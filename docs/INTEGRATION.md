# Integration with audio-pipeline

Preferred product chain:

```text
microphones -> audio-pipeline (BF/AEC/RES/NS/AGC as SKU requires)
            -> mono PCM16 16 kHz
            -> kws-pipeline
            -> wake event
            -> higher-level ASR/assistant session
```

Do not run two independent resamplers when `audio-pipeline` already emits 16-kHz mono. KWS normally consumes post-AEC/post-NS audio so local speaker playback and device noise are reduced before wake detection. Validate thresholds against the exact AGC configuration as aggressive gain changes alter score distributions.

## Field-updatable initialization

```c
kws_model_t model;
kws_keyword_pack_t pack;
kws_engine_t *kws;
kws_config_t cfg = kws_default_config();

kws_model_open(model_blob, model_blob_bytes, &model);
kws_keyword_pack_open(pack_blob, pack_blob_bytes, &model, &pack);
size_t bytes = kws_engine_required_bytes(&model);
size_t alignment = kws_engine_required_alignment();
/* Allocate/provide `arena` with at least `bytes` and `alignment`. */
kws_engine_init(arena, bytes, &model, &cfg, &kws);
kws_engine_set_keyword_pack(kws, &pack);
```

ABI v2 requires the `.kwm` and `.kwk` vocabulary fingerprints to match exactly. The keyword pack object may be discarded after the setter returns; the model blob must remain alive because model tensors are zero-copy views into it. The model blob must also be at least float-aligned.

For a firmware-linked generated C table use:

```c
kws_engine_set_keywords(kws,
                        kws_generated_keywords,
                        kws_generated_keyword_count,
                        kws_generated_vocab_fingerprint);
```

The C-table path performs the same vocabulary identity check as `.kwk`.

## Audio calls

ABI v2 fixes 16-kHz, 400-sample frames and 320-sample hops. Each call to `kws_engine_accept_pcm16()` accepts at most `KWS_MAX_PCM_BLOCK_SAMPLES` (320) samples. Inputs larger than one hop return `KWS_EBOUNDS` without consuming samples.

The recommended integration is to pass each normal `audio-pipeline` 10-ms output block directly:

```text
160 samples @ 16 kHz -> kws_engine_accept_pcm16(...)
```

This one-hop upper bound guarantees one call can produce at most one acoustic step, so the single `kws_detection_t` output cannot silently discard multiple wake events. Accepted blocks are always consumed completely, including a block in which a wake event occurs.

One engine is single-owner. If capture and assistant state machines run on different threads, publish only the small wake event across a queue rather than calling one engine concurrently.

## Runtime ownership

- The caller owns the `.kwm` bytes and engine arena.
- The real-time library performs no file I/O or heap allocation.
- File loading in `kws_wav` belongs only to the hosted evaluation tool.
- Keyword updates belong on the control path, not from concurrent audio callbacks.
- A rejected keyword update leaves the previous valid trie active.
- `kws_engine_reset()` resets acoustic/decoder/refractory state but keeps the configured keyword trie and monotonic processed-sample position.

Threshold qualification must use the exact final `audio-pipeline` stage composition and gain policy shipped on the SKU.
