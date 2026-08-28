# Architecture

`kws-pipeline` separates the offline/control plane from the always-on real-time plane.

## Real-time data path

```text
mono PCM16 @ 16 kHz
 -> fixed 25-ms Hann / 20-ms hop
 -> 512 FFT
 -> 32-bin log-mel + per-frame mean normalization
 -> int8-weight tiny recurrent acoustic model
 -> blank + pinyin token logits
 -> shared-prefix keyword trie
 -> speech gate + threshold + refractory gate
 -> detection {keyword_id, confidence, end_sample}
```

ABI v2 fixes the acoustic geometry to 400-sample frames and 320-sample hops. A call to `kws_engine_accept_pcm16()` may contain at most `KWS_MAX_PCM_BLOCK_SAMPLES` (320) samples. This guarantees that one call can produce at most one acoustic step, matching the API's single detection output while still allowing the engine to consume the complete accepted block. The preferred `audio-pipeline` integration uses its normal 10-ms / 160-sample output blocks.

The runtime is C11, uses no heap, owns no worker thread, does no filesystem I/O and performs no Chinese text/pinyin conversion. The caller supplies one aligned arena and keeps the model blob alive for the engine lifetime.

## Open-token KWS

A fixed-word binary classifier is cheap but normally needs a new model for every phrase. This engine instead learns reusable pinyin acoustics and decodes only configured token paths. A phrase such as `你好小窝` (`ni3 hao3 xiao3 wo1`) can therefore be introduced without retraining when the generic acoustic model already separates its tokens well enough.

The default 32-feature / 48-hidden / ~420-token geometry is about 1.2 MMAC/s for the dense acoustic network and about 26 KB for weights+biases. These are design calculations, not target-board measurements.

## Ownership contract

- model blob: caller-owned, read-only, stable address and at least float-aligned because model biases are zero-copy float views;
- engine arena: caller-owned, exclusive to one engine, with at least `kws_engine_required_alignment()` alignment;
- keyword arrays: required only during `kws_engine_set_keywords()`; the internal trie retains token IDs, not caller pointers;
- parsed keyword pack: required only until `kws_engine_set_keyword_pack()` returns;
- engine: single-thread owner; serialize externally;
- no global mutable state, heap, locks or runtime plugins.

`kws_engine_required_bytes()` returns zero for a model that violates the public model contract. `kws_engine_init()` repeats geometry, pointer, finite-float and config checks even when a caller manually constructs the public `kws_model_t` rather than using `kws_model_open()`.

## Hard bounds

Current limits are 16 keywords, 16 tokens per keyword, 40 features, 64 recurrent units and 512 acoustic tokens. Model/keyword ABI v2 additionally fixes 16-kHz input, 400-sample analysis frames and 320-sample acoustic hops. These are product bounds that keep resident memory and per-call event semantics deterministic.
