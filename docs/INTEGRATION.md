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
kws_engine_init(arena, bytes, &model, &cfg, &kws);
kws_engine_set_keyword_pack(kws, &pack);
```

ABI v2 requires the `.kwm` and `.kwk` vocabulary fingerprints to match exactly. The keyword pack object may be discarded after the setter returns; the model blob must remain alive because model tensors are zero-copy views into it.

For a firmware-linked generated C table use:

```c
kws_engine_set_keywords(kws,
                        kws_generated_keywords,
                        kws_generated_keyword_count,
                        kws_generated_vocab_fingerprint);
```

The C-table path performs the same vocabulary identity check as `.kwk`.

Feed every mono PCM16 block to `kws_engine_accept_pcm16()`. Chunk size is arbitrary; the engine internally produces 20-ms acoustic steps after the first 25-ms frame. A call consumes the complete PCM block even when a wake event is detected partway through it.

One engine is single-owner. If capture and assistant state machines run on different threads, publish only the small wake event across a queue rather than calling one engine concurrently.

## Runtime ownership

- The caller owns the `.kwm` bytes and engine arena.
- The real-time library performs no file I/O or heap allocation.
- File loading in `kws_wav` belongs only to the hosted evaluation tool.
- Keyword updates belong on the control path, not from concurrent audio callbacks.
- A rejected keyword update leaves the previous valid trie active.

Threshold qualification must use the exact final `audio-pipeline` stage composition and gain policy shipped on the SKU.
