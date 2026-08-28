# Architecture

`kws-pipeline` separates the offline/control plane from the always-on real-time plane.

## Real-time data path

```text
mono PCM16 @ 16 kHz
 -> 25-ms Hann / 20-ms hop
 -> 512 FFT
 -> 32-bin log-mel + per-frame mean normalization
 -> int8-weight tiny recurrent acoustic model
 -> blank + pinyin token logits
 -> shared-prefix keyword trie
 -> speech gate + threshold + refractory gate
 -> detection {keyword_id, confidence, end_sample}
```

The runtime is C11, uses no heap, owns no worker thread, does no filesystem I/O and performs no Chinese text/pinyin conversion. The caller supplies one aligned arena and keeps the model blob alive for the engine lifetime.

## Open-token KWS

A fixed-word binary classifier is cheap but normally needs a new model for every phrase. This engine instead learns reusable pinyin acoustics and decodes only configured token paths. A phrase such as `你好小窝` (`ni3 hao3 xiao3 wo1`) can therefore be introduced without retraining when the generic acoustic model already separates its tokens well enough.

The default 32-feature / 48-hidden / ~420-token geometry is about 1.2 MMAC/s for the dense acoustic network and about 26 KB for weights+biases. These are design calculations, not target-board measurements.

## Ownership contract

- model blob: caller-owned, read-only, stable address;
- engine arena: caller-owned, aligned, exclusive to one engine;
- keyword arrays: required only during `kws_engine_set_keywords()`; the internal trie retains token IDs, not caller pointers;
- engine: single-thread owner; serialize externally;
- no global mutable state, heap, locks or runtime plugins.

## Hard bounds

Current limits are 16 keywords, 16 tokens per keyword, 40 features, 64 recurrent units and 512 acoustic tokens. They are product bounds that keep resident memory deterministic.
