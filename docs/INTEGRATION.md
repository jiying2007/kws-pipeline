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

The current hard-cut artifact contract is **KWSP model ABI v2 + KWKP keyword-pack ABI v3**. Their vocabulary fingerprints must match exactly. The keyword pack object may be discarded after the setter returns; the model blob must remain alive because model tensors are zero-copy views into it. The model blob must also be at least float-aligned.

For a firmware-linked generated C table use:

```c
kws_engine_set_keywords(kws,
                        kws_generated_keywords,
                        kws_generated_keyword_count,
                        kws_generated_vocab_fingerprint);
```

The C-table path performs the same vocabulary identity check as `.kwk`.

For overlapping keywords, do not treat each phrase as an independent detector. Compile the shared token paths into the same KWKP v3 pack and use `min_trailing_blanks`, `priority`, `prefix_policy` and `grace_frames`. `keywords/zh_cn_overlap_example.tsv` shows the short/long pair `小窝` / `小窝小窝`.

## Audio calls

KWSP v2 fixes 16-kHz, 400-sample frames and 320-sample hops. Each call to `kws_engine_accept_pcm16()` accepts at most `KWS_MAX_PCM_BLOCK_SAMPLES` (320) samples. Inputs larger than one hop return `KWS_EBOUNDS` without consuming samples.

The recommended integration is to pass each normal `audio-pipeline` 10-ms output block directly:

```text
160 samples @ 16 kHz -> kws_engine_accept_pcm16(...)
```

Products with capture metadata should use `kws_engine_accept_pcm16_ex()`. The versioned metadata carries
stream sequence/timestamp, discontinuity/XRUN/codec-reopen/clock-reset flags, lost samples, optional
external AFE VAD probability, AFE latency and the exact AFE configuration SHA-256. A discontinuity
deterministically clears frontend/RNN/decoder/refractory state without clearing keywords or monotonic
telemetry. Passing `sample_count == 0` is valid for a metadata-only discontinuity notification.

When `KWS_FRAME_EXTERNAL_VAD_VALID` is present, the final AFE VAD is authoritative for that block;
otherwise KWS falls back to its configured post-AGC dBFS gate. Do not mix unrelated VAD and PCM timelines.

This one-hop upper bound guarantees one call can produce at most one acoustic step, so the single `kws_detection_t` output cannot silently discard multiple wake events. Accepted blocks are always consumed completely, including a block in which a wake event occurs.

One engine is single-owner. If capture and assistant state machines run on different threads, publish only the small wake event across a queue rather than calling one engine concurrently.

## Zero-I/O telemetry

The real-time library deliberately has no logging or dump path. Product diagnostics can snapshot bounded counters without filesystem or heap activity:

```c
kws_engine_stats_t stats;
if (kws_engine_get_stats(kws, &stats) == KWS_OK) {
    /* Publish/copy on your control path; do not printf from the audio callback. */
}
```

The original snapshot exposes processed/speech/blank frames, decoder hits, refractory suppressions,
accepted detections, current keyword/Trie size, pending prefix state and maximum detection confidence.
`kws_engine_get_stats_v2()` additionally exposes discontinuity/lost-sample/external-VAD counters and the
last AFE identity/timing metadata. Counters are cumulative for the engine lifetime; `kws_engine_reset()`
resets acoustic/decoder/refractory state but intentionally keeps monotonic runtime telemetry.

Log `kws_build_info()` together with the model/keyword hashes. It binds runtime version, source revision,
compiler, target triple, build type and configuration digest.

## Runtime ownership

- The caller owns the `.kwm` bytes and engine arena.
- The real-time library performs no file I/O or heap allocation.
- File loading in `kws_wav` belongs only to the hosted evaluation tool.
- Keyword updates belong on the control path, not from concurrent audio callbacks.
- A rejected keyword update leaves the previous valid trie active.
- `kws_engine_reset()` resets acoustic/decoder/refractory state but keeps the configured keyword trie and monotonic processed-sample/statistics counters.

Threshold qualification must use the exact final `audio-pipeline` stage composition and gain policy shipped on the SKU.

## Shipping AFE adapter contract

Synthetic/measured-domain training can invoke the final `audio-pipeline` through the command AFE backend. The command must write both `{output}` and `{result}`. The result sidecar must contain `latency_samples`; it may additionally report `pipeline_sha256`, `source_sha` and `toolchain`.

AFE identity is not derived from temporary file names. It binds the command template, the actual executable SHA256, declared config-file bundle SHA256, input/output hashes and result sidecar. The reported latency is added to event references so BF/AEC/NS buffering or lookahead cannot silently shift wake-latency scoring.
