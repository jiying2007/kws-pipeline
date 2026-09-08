# Integration with audio-pipeline

Preferred product chain:

```text
microphones -> audio-pipeline (BF/AEC/RES/NS/AGC as SKU requires)
            -> mono PCM16 16 kHz
            -> kws-pipeline
            -> wake event
            -> higher-level ASR/assistant session
```

Do not run two independent resamplers when `audio-pipeline` already emits 16-kHz mono. KWS normally consumes post-AEC/post-NS audio so local speaker playback and device noise are reduced before wake detection. Validate thresholds against the exact AGC configuration because gain changes alter score distributions.

The current promoted model is `model-749187ec1d66`. `configs/shipping.xiaowo.json` is the machine-readable product authority and contains exactly `你好小窝` and `小窝小窝`; generic overlap fixtures are decoder tests, not additional shipping wake words.

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

The deployable runtime artifact contract remains **KWSP model ABI v2 + KWKP keyword-pack ABI v3**. Their vocabulary fingerprints must match exactly. The keyword-pack object may be discarded after the setter returns; the model blob must remain alive because model tensors are zero-copy views into it and must satisfy the documented alignment requirement.

For a firmware-linked generated C table use:

```c
kws_engine_set_keywords(kws,
                        kws_generated_keywords,
                        kws_generated_keyword_count,
                        kws_generated_vocab_fingerprint);
```

The generated-table path performs the same vocabulary identity check as `.kwk`.

For overlapping phrases, KWKP v3 supports `min_trailing_blanks`, `priority`, `prefix_policy` and `grace_frames`. `keywords/zh_cn_overlap_example.tsv` is a **decoder contract fixture only**; `keywords/zh_cn_example.tsv` is the current shipping pack.

## Audio calls

KWSP v2 fixes 16-kHz, 400-sample frames and 320-sample hops. Each call to `kws_engine_accept_pcm16()` accepts at most `KWS_MAX_PCM_BLOCK_SAMPLES` (320) samples. Inputs larger than one hop return `KWS_EBOUNDS` without consuming samples.

The recommended integration is to pass each normal 10-ms output block directly:

```text
160 samples @ 16 kHz -> kws_engine_accept_pcm16(...)
```

The one-hop upper bound guarantees one call can produce at most one acoustic step, so the single `kws_detection_t` output cannot silently discard multiple wake events. Accepted blocks are consumed completely.

One engine is single-owner. If capture and assistant state machines run on different threads, publish only the small wake event across a queue rather than calling one engine concurrently.

## Mandatory discontinuity handling

A capture timeline is not always continuous. XRUN, USB/I2S route changes, sample-clock resets and suspend/resume can drop or skip samples. Do **not** feed post-gap PCM into the previous acoustic state as if no gap occurred.

Call:

```c
kws_engine_notify_discontinuity(kws, KWS_DISCONTINUITY_XRUN);
```

or the matching reason:

- `KWS_DISCONTINUITY_XRUN`;
- `KWS_DISCONTINUITY_ROUTE_CHANGE`;
- `KWS_DISCONTINUITY_CLOCK_RESET`;
- `KWS_DISCONTINUITY_SUSPEND_RESUME`.

The discontinuity API clears partial frontend framing, PCEN smoothing, RNN hidden state, decoder/pending-prefix state and refractory suppression. It intentionally preserves configured keywords and monotonic telemetry. This prevents a partial wake phrase before a gap from being structurally joined to unrelated audio after the gap.

Do not use ordinary `kws_engine_reset()` as a substitute for product timeline semantics unless a complete logical KWS session reset is intended. See `docs/AUDIO_DISCONTINUITY.md`.

## Zero-I/O telemetry

The real-time library deliberately has no logging or dump path. Product diagnostics can snapshot bounded counters without filesystem or heap activity:

```c
kws_engine_stats_t stats;
if (kws_engine_get_stats(kws, &stats) == KWS_OK) {
    /* Publish/copy on the control path; do not printf from the audio callback. */
}
```

The snapshot includes processed/speech/blank frames, decoder hits, refractory suppressions, detections, configured keyword/Trie size, pending prefix state, maximum confidence and discontinuity counters. Counters are cumulative for the engine lifetime.

For long product soaks, copy stats on a non-real-time control path and combine them with audio-pipeline XRUN/backpressure counters. Qualification should retain both KWS and upstream audio continuity evidence.

## Runtime ownership

- The caller owns the `.kwm` bytes and engine arena.
- The real-time library performs no file I/O or heap allocation.
- File loading in `kws_wav` belongs only to hosted/evaluation tools.
- Keyword updates belong on the control path, not from concurrent audio callbacks.
- A rejected keyword update leaves the previous valid trie active.
- `kws_engine_reset()` resets acoustic/decoder/refractory state while retaining configured keywords and monotonic lifetime telemetry.
- `kws_engine_notify_discontinuity()` is the product-facing boundary for capture timeline breaks.

## Shipping AFE adapter contract

Synthetic/measured-domain training can invoke the final `audio-pipeline` through the command AFE backend. The command must write both `{output}` and `{result}`. The result sidecar must contain `latency_samples`; it may additionally report `pipeline_sha256`, `source_sha` and `toolchain`.

AFE identity binds the command template, actual executable SHA256, declared config-file bundle SHA256, input/output hashes and result sidecar rather than temporary paths. Reported AFE latency is added to event timing so BF/AEC/NS buffering or lookahead cannot silently shift wake-latency scoring.

The repository-side shipping schema is **`commercial/afe-evidence.schema.json`**. Shipping-authoritative AFE evidence must bind at least:

- `backend=command` and `shipping_authority=true`;
- exact executable and config-bundle SHA256;
- input PCM, output PCM and result-sidecar SHA256;
- non-negative `latency_samples`;
- mono PCM16 16 kHz;
- exact SKU, microphone revision and enclosure revision.

The synthetic `proxy` backend remains valid for development/domain stress, but it has no shipping authority. It cannot clear `shipping_evidence_boundary.final_afe_required` in `configs/shipping.xiaowo.json`.

Threshold qualification must use the exact final microphone/enclosure/audio-pipeline stage composition and gain policy shipped on the SKU. Changing the AFE executable/config or hardware sound path creates a new qualification tuple.

## Commercial-candidate deployment identity

`.github/workflows/deployment-release.yml` creates a content-addressed `deployment-<source-sha12>` candidate only from the exact current protected `main`. It refuses publication unless:

- the immutable promoted Model Release still targets exact trained HEAD `749187ec1d6662658f06aa9c76d47fde835968db`;
- current runtime source (`CMakeLists.txt`, `cmake/`, `include/`, `src/`) is unchanged from the source used by the qualified model;
- every promoted-model asset in `MODEL_SHA256SUMS` verifies;
- two independently built installed SDK trees are byte-for-byte reproducible;
- SDK/source/SPDX/model/qualification/robustness/FAR/shipping/nightly/AFE identities are frozen into `deployment-manifest.json` and `DEPLOYMENT_SHA256SUMS`;
- provenance and SDK-SBOM attestations succeed.

The deployment manifest is deliberately `commercial-candidate` with `shipping_approved=false`. Its only remaining blockers are the two evidence classes already declared by `configs/shipping.xiaowo.json`: real-human final-AFE acoustic qualification and physical target-board performance/soak.

## Release evidence boundary

The final product tuple must retain enough identity to connect the shipping data path to the candidate deployment and final qualification:

- exact deployment Release/source SHA and SDK identity;
- exact KWS model/pack/config;
- exact audio-pipeline executable/config identity and schema-valid AFE evidence;
- microphone/enclosure/device revision;
- discontinuity/XRUN policy and soak counters;
- representative post-AFE board benchmark audio;
- original held-out human qualification WAVs and corpus identity;
- physical target-board raw measurements.

Hosted tests can prove API and evidence contracts. Only the final device/audio path can prove real FAR/FRR, far-field behavior and physical performance. Until those two real-world evidence classes pass, `shipping_approved` must remain false.
