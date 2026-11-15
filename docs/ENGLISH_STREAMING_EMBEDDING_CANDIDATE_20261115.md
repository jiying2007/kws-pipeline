# English streaming speech-embedding candidate — 2026-11-15

Status: architecture/source review only. No feature assets downloaded, no inference, no training, no model promotion, and no SSC305 performance claim.

## Why this candidate follows the failed English student experiments

The retained GRU and DS-TCN classifiers used whole-clip mean/max pooling. Their encoder chunk parity did not make the classifier an online event detector. Their later continuous C-event experiments also failed the fixed joint gate: background suppression was achieved at the cost of one or both keyword-positive coverage conditions.

The next architecture should not simply continue adjusting the same clip-level objective. A promising distinct route is a frozen, general-purpose speech embedding backbone with a small causal per-frame event head, trained on source-disjoint positive and negative windows.

## Exact upstream reference

Project: https://github.com/dscripka/openWakeWord

Reviewed source revision: `368c03716d1e92591906a84949bc477f3a834455` (2025-12-30).

The upstream README describes three components:
1. fixed 16-kHz audio-to-log-mel preprocessing;
2. a shared speech embedding extractor based on Google's `speech_embedding/1`, declared Apache-2.0;
3. a small wake-word classifier.

The official upstream `v0.5.1` release exposes the following feature assets. Asset IDs and byte sizes are recorded here for exact retrieval; SHA-256 remains unmeasured and must be computed from the downloaded bytes before any use.

| Asset | GitHub release asset ID | Bytes | Role |
|---|---:|---:|---|
| `melspectrogram.tflite` | 129656230 | 1,092,516 | 16-kHz audio to 32-band mel features |
| `embedding_model.tflite` | 129656244 | 1,330,312 | 76×32 mel window to 96-dimensional embedding |
| `melspectrogram.onnx` | 129656233 | 1,087,958 | Alternative frontend graph |
| `embedding_model.onnx` | 129656245 | 1,326,578 | Alternative embedding graph |

Release tag: `v0.5.1`. URLs:
- https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite
- https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite
- https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx
- https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx

The upstream repository code and Google speech-embedding feature model are Apache-2.0. The distributed openWakeWord pretrained wake-word classifier models are CC BY-NC-SA 4.0 and are excluded from the product candidate. No third-party negative corpus, classifier checkpoint, or noncommercial model is admitted by this review.

## Streaming geometry

The reviewed upstream implementation uses:
- 16-kHz mono input;
- 25-ms analysis window and 10-ms mel hop;
- 32 mel bands;
- 76 mel frames per embedding input (760 ms of trailing acoustic context);
- one 96-dimensional embedding every 8 mel frames (80 ms);
- a rolling sequence of 16 embeddings for the default classifier input (1.28 s of embedding history).

The 760-ms feature window is trailing context, not 760 ms of future lookahead. Once the history is warm, a new embedding can be produced every 80 ms. A causal event head can score the latest rolling embedding window on that cadence. Initial warm-up, model compute, buffering, and event confirmation still need to be measured; this geometry does not itself prove the 500-ms word-end callback target.

The 16×96 head input is a reference geometry, not a frozen choice for this project. A shorter context may reduce head compute but could lose phrase evidence. Any 8/16-frame comparison must be predeclared, source-identical and evaluated with the same decoder; no result-driven context sweep.

## Proposed candidate

- Frozen shared speech embedding backbone.
- Small causal classifier head over the latest embedding history.
- Separate logits for Hey Mero, Hey Vosen, and background.
- Per-frame output every 80 ms, with a single shared frontend/backbone for both phrases.
- Existing continuous event scorer and explicit reset/discontinuity contract.
- No whole-clip mean/max pooling and no bidirectional recurrent head.
- Train-only normalization and source-family-separated train/development/calibration/locked confirmation.

A small DNN head is the first reference because it is straightforward to export and implement in C. A causal FSMN-CTC remains a separate challenger, not an assumed winner. Do not copy the upstream two-layer bidirectional LSTM as a streaming head.

## Required asset qualification before implementation

1. Download only the four exact release assets above from the official release; record response metadata, byte counts and SHA-256. Do not rely on the release tag alone after retrieval.
2. Inspect TFLite/ONNX operators, tensor shapes, input scaling, quantization, padding, receptive field, and batch/stream behavior.
3. Compare TFLite and ONNX outputs on invented fixtures, then source-bound real inputs only after pronunciation review.
4. Determine whether SSC305 has a compatible runtime. If not, implement the exact graph in native C with a versioned layout; do not assume the presence of TFLite or ONNX Runtime in the target rootfs.
5. Measure the complete frontend + embedding + head, not only the classifier.
6. Test rolling-window parity under arbitrary PCM chunk sizes, startup warm-up, long streams, gaps, resets, and end-of-stream. No silent padding or per-clip state reset unless explicitly part of the protocol.

The two TFLite files total 2,422,828 bytes before any model head or runtime overhead. This exceeds the V4 preferred 2-MiB model-size target if counted together. Quantization or a smaller implementation may be required, but no size reduction is assumed until the exact asset operators and quantization are inspected.

## Data and pronunciation boundary

Owner-confirmed target readings:
- Hey Mero: /heɪ ˈmɪəroʊ/
- Hey Vosen: /heɪ ˈvoʊsən/

The previous 63 seven-voice TTS WAVs were generated with Qwen3-TTS 0.6B CustomVoice. The official 0.6B CustomVoice implementation clears per-utterance `instruct`; therefore those clips were not conditioned by these newly confirmed IPA instructions. They remain immutable and exposed, not relabelled.

The four-cell Qwen3-TTS 1.7B pronunciation probe is a separate TTS feasibility step. Its output must be acoustically reviewed before any sample is admitted as a positive. Synthetic preset voices are not independent human speakers. The 517 LibriSpeech clips and all previously inspected English63 clips remain exposed research data, not unseen qualification.

## Frozen completion gates

For the same final model + frontend + decoder + threshold configuration:
- combined two-keyword FAR <= 0.1/hour;
- each keyword FRR <= 5%;
- one-sided 95% upper confidence bounds reported;
- word-end to consumable callback P95 <= 500 ms;
- C/ARM numerical and event parity;
- actual SSC305 BSP/audio-chain CPU, RAM, latency, thermal and long-run measurements.

No model, dataset, source asset, runtime dependency, or training recipe is admitted by this document. The next executable stage is exact asset retrieval and operator inspection, followed by a source-bound streaming feature parity check. Model training waits for reviewed pronunciation-specific positives and source-disjoint negative coverage.
