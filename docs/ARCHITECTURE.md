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
 -> dominant-token CTC admission
 -> shared-prefix keyword trie
 -> speech gate + threshold + refractory gate
 -> detection {keyword_id, confidence, end_sample}
```

ABI v2 fixes the acoustic geometry to 400-sample frames and 320-sample hops. A call to `kws_engine_accept_pcm16()` may contain at most `KWS_MAX_PCM_BLOCK_SAMPLES` (320) samples. This guarantees that one call can produce at most one acoustic step, matching the API's single detection output while still allowing the engine to consume the complete accepted block. The preferred `audio-pipeline` integration uses its normal 10-ms / 160-sample output blocks.

The runtime is C11, uses no heap, owns no worker thread, does no filesystem I/O and performs no Chinese text/pinyin conversion. The caller supplies one aligned arena and keeps the model blob alive for the engine lifetime.

## Open-token KWS

A fixed-word binary classifier is cheap but normally needs a new model for every phrase. This engine instead learns reusable pinyin acoustics and decodes only configured token paths. A phrase such as `你好小窝` (`ni3 hao3 xiao3 wo1`) can therefore be introduced without retraining when the generic acoustic model already separates its tokens well enough.

The default 32-feature / 48-hidden / ~420-token geometry is about 1.2 MMAC/s for the dense acoustic network and about 26 KB for weights+biases. These are design calculations, not target-board measurements.

## Keyword decoder state model

The decoder is a bounded shared-prefix Trie with a lightweight Viterbi-style path scorer. It is intentionally not a general-purpose CTC prefix-beam search.

Before a Trie edge may advance, the frame is reduced to one **admitted CTC label**: the highest-logit nonblank token, but only when that token also beats blank. A blank-dominant frame admits no nonblank token. Other non-dominant token posteriors still contribute to the frame normalization/confidence denominator, but they cannot advance keyword structure.

This dominant-token admission rule is deliberate. Allowing every Trie child with any nonzero posterior to advance lets transition-frame tails plus `token_boost` fabricate missing or reordered labels. For example, a strongly decoded `ni3 hao3 wo1 xiao3` sequence must not complete `ni3 hao3 xiao3 wo1` merely because `xiao3`/`wo1` had weak secondary posterior on neighboring frames.

Each non-root Trie node retains two independent scores:

- `score`: best prefix path whose latest token has not subsequently been separated by a blank;
- `blank_score`: best prefix path after at least one blank-dominant acoustic frame since the latest emitted token.

A non-repeated child can advance from the better of those two states, but only when that child is the admitted dominant label for the frame. If the child token is identical to the current node token, it can advance **only** from `blank_score`. This mirrors the structural CTC rule that adjacent identical target labels require a blank separator without carrying a general beam over the entire vocabulary.

Blank and nonblank scores must remain separate. Collapsing them into one score plus a boolean would either discard a lower-scoring but valid separated path or incorrectly grant blank readiness to a higher-scoring unseparated path.

State retention remains a product-oriented gap heuristic rather than exact CTC probability accumulation: a blank-dominant frame moves retained nonblank state into the separated state, while an already separated state remains separated until another token is emitted. An unrelated dominant token does not destroy an existing partial prefix; it simply cannot advance an incompatible edge. This keeps the decoder bounded and inexpensive while preserving the repeated-label constraint and ordered-token contract.

## Frontend contract

The device frontend and training frontend are treated as one versioned feature contract. `training/frontend_spec.py` defines the dependency-free reference geometry and constants; `kws_feature_dump` runs the actual C implementation on WAV input; CI compares C output frame-by-frame against the reference with a tight float tolerance. The PyTorch frontend imports the same FFT/mel/scale constants from the reference module.

A frontend geometry, windowing, mel-bin or normalization change therefore requires an explicit contract update rather than silently changing training features independently of the embedded runtime.

## Training provenance

Training does not accept an unqualified vocabulary size. `training/train_ctc.py` requires the actual token vocabulary and stores its 64-bit fingerprint plus token-file SHA256 in the checkpoint. Warm starts verify the fingerprint before loading weights, and `training/export_model.py` refuses to bind a checkpoint to a same-sized but differently mapped vocabulary.

Checkpoints additionally retain training-manifest hashes, frontend-spec version, seed and optimizer settings so the acoustic artifact has a reproducible lineage before `.kwm` export.

The dependency-free synthetic CI backend also emits explicit fitting provenance. Its competitive softmax acoustic head is trained only from deterministic token/background fitting samples, quantized to the same ABI-v2 int8 format used by the C runtime, and then re-evaluated after quantization. CI requires the held-out token-fit validation accuracy to remain at least 99.5% before the candidate can participate in the end-to-end synthetic gate.

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

Parser attack-surface hardening is exercised by deterministic contract tests plus Clang libFuzzer/ASan/UBSan smoke jobs for `.kwm` and `.kwk` inputs. Fuzzing is a software safety gate; it does not replace authentication/signing of production update artifacts.
