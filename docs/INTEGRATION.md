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

Typical initialization:

```c
kws_model_t model;
kws_engine_t *kws;
kws_config_t cfg = kws_default_config();

kws_model_open(model_blob, model_blob_bytes, &model);
size_t bytes = kws_engine_required_bytes(&model);
kws_engine_init(arena, bytes, &model, &cfg, &kws);
kws_engine_set_keywords(kws, kws_generated_keywords,
                        kws_generated_keyword_count);
```

Feed every mono PCM16 block to `kws_engine_accept_pcm16()`. Chunk size is arbitrary; the engine internally produces 20-ms acoustic steps after the first 25-ms frame.

One engine is single-owner. If capture and assistant state machines run on different threads, publish only the small wake event across a queue rather than calling one engine concurrently.
