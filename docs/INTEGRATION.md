# Integration with audio-pipeline

> **Current authority: technical research only.** The frozen synthetic model and keyword source is `configs/research/two-keyword-synthetic-comparator.json`; it is not real-human FAR/FRR, a physical SSC305 measurement or commercial publication authority. Old product/Release terminology elsewhere on this page is historical context, not a workflow to execute.

Preferred product chain:

```text
microphones -> audio-pipeline (BF/AEC/RES/NS/AGC as SKU requires)
            -> mono PCM16 16 kHz
            -> kws-pipeline
            -> wake event
            -> higher-level ASR/assistant session
```

Do not run two independent resamplers when `audio-pipeline` already emits 16-kHz mono. KWS normally consumes post-AEC/post-NS audio so local speaker playback and device noise are reduced before wake detection. Validate thresholds against the exact AGC configuration because gain changes alter score distributions.

`model-749187ec1d66` is a frozen historical synthetic comparator, not an approved current model. `configs/research/two-keyword-synthetic-comparator.json` pins the exact `你好小窝` and `小窝小窝` research wake phrases and original SHA256s; generic overlap fixtures are decoder tests, not additional measured phrases.

## Field-updatable initialization

The caller supplies a separate candidate arena aligned to
`kws_engine_required_alignment()`. Keep the active engine and its arena intact
until all candidate initialization steps succeed. This helper leaves the output
pointer unchanged on failure:

```c
kws_status_t prepare_kws_candidate(const void *model_blob, size_t model_blob_bytes,
                                  const void *pack_blob, size_t pack_blob_bytes,
                                  void *arena, size_t arena_bytes,
                                  kws_engine_t **out_candidate) {
  kws_model_t model = {0};
  kws_keyword_pack_t pack = {0};
  kws_engine_t *candidate = NULL;
  kws_config_t cfg = kws_default_config();
  kws_status_t status;

  if (out_candidate == NULL) {
    return KWS_EINVAL;
  }
  status = kws_model_open(model_blob, model_blob_bytes, &model);
  if (status != KWS_OK) {
    return status;
  }
  status = kws_keyword_pack_open(pack_blob, pack_blob_bytes, &model, &pack);
  if (status != KWS_OK) {
    return status;
  }
  if (arena_bytes < kws_engine_required_bytes(&model)) {
    return KWS_ENOMEM;
  }
  status = kws_engine_init(arena, arena_bytes, &model, &cfg, &candidate);
  if (status != KWS_OK) {
    return status;
  }
  status = kws_engine_set_keyword_pack(candidate, &pack);
  if (status != KWS_OK) {
    return status;
  }
  *out_candidate = candidate;
  return KWS_OK;
}
```

After `KWS_OK`, publish the candidate under the caller's existing single-owner
synchronization. On failure, keep the active engine and reject the candidate;
zero-initializing the temporary structures does not replace checking statuses.

The deployable runtime artifact contract remains **KWSP model ABI v2 + KWKP keyword-pack ABI v3**. Their vocabulary fingerprints must match exactly. The keyword-pack object may be discarded after the setter returns; the model blob must remain alive because model tensors are zero-copy views into it and must satisfy the documented alignment requirement.

For a firmware-linked generated C table use:

```c
kws_status_t status = kws_engine_set_keywords(
    kws, kws_generated_keywords, kws_generated_keyword_count,
    kws_generated_vocab_fingerprint);
if (status != KWS_OK) {
  /* Report the rejected update; the previous valid keyword configuration remains. */
  return status;
}
```

The generated-table path performs the same vocabulary identity check as `.kwk`.

For overlapping phrases, KWKP v3 supports `min_trailing_blanks`, `priority`, `prefix_policy` and `grace_frames`. `keywords/zh_cn_overlap_example.tsv` is a **decoder contract fixture only**; `keywords/zh_cn_example.tsv` is the current shipping pack.

## Audio calls

KWSP v2 fixes 16-kHz, 400-sample frames and 320-sample hops. Each call to `kws_engine_accept_pcm16()` accepts at most `KWS_MAX_PCM_BLOCK_SAMPLES` (320) samples. Inputs larger than one hop return `KWS_EBOUNDS` without consuming samples.

The recommended integration is to pass each normal 10-ms output block directly:

```text
160 samples @ 16 kHz -> kws_engine_accept_pcm16(...)
```

The one-hop upper bound guarantees one call can produce at most one acoustic step, so the single `kws_detection_t` output cannot silently discard multiple wake events. Accepted blocks are consumed completely. The 10-ms callback cadence is distinct from the 20-ms acoustic hop: measure the actual callback scheduling/work budget on the product. `kws_board_bench` times calls of up to 320 samples (the final call may be shorter) against a nominal 20-ms block deadline; its p99 headroom does not qualify a 10-ms callback deadline.

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

The research source-only AFE schema is **`configs/research/afe-evidence.schema.json`** and the adapter contract is **`configs/research/final-afe-adapter.schema.json`**. These bind measurement provenance, not product qualification or publication authority. An actual command-AFE experiment must retain at least:

- `backend=command` and `research_measurement_only=true` (old `shipping_authority` is rejected);
- exact executable and config-bundle SHA256;
- input PCM, output PCM and result-sidecar SHA256;
- non-negative `latency_samples`;
- mono PCM16 16 kHz;
- exact SKU, microphone revision and enclosure revision.

The synthetic `proxy` backend remains useful for development/domain stress but cannot become an independently recorded human/final-AFE measurement. It cannot replace real external microphone/enclosure/AFE evidence.

Independent threshold research must bind the exact tested microphone/enclosure/audio-pipeline composition, executable/config hash and acoustic route. Changing those inputs creates a new experimental identity; no commercial shipping permission follows.

## Historical deployment identity (retired)

The previous `deployment-release.yml` published content-addressed commercial
candidate SDK/model/evidence archives after checking exact protected-main
source, model hashes, reproducible SDK builds, and provenance. That
**publication-only workflow has been removed from active Actions**. Its
original source and receipts are preserved in Git/Actions history, not
part of current research permissions or experiment execution.

For current research, pin and compare the real model/keyword pack, CMake
configuration/source, generated parameter header inputs, saved PCM,
evaluation labels and source families; use the existing scientific CI
and controlled dataset iteration, not a shipping configuration. Physical
SSC305 and real-human AFE results are independent measurements and remain
NOT_RUN unless they have actually been collected.

The separate research lane also retains unresolved numerical/data-admission
findings: D20 **FAIL**, D90 **NOT_RUN**, and frozen Fixed50
**EVAL_INCONCLUSIVE_LABEL_SUPPORT**. Historical source-bound results cannot
be upgraded by a release tag or a workflow name.

## Historical Release evidence boundary (not active research admission)

The final product tuple must retain enough identity to connect the shipping data path to the candidate deployment and final qualification:

- exact deployment Release/source SHA and SDK identity;
- exact KWS model/pack/config;
- exact audio-pipeline executable/config identity and schema-valid AFE evidence;
- microphone/enclosure/device revision;
- discontinuity/XRUN policy and soak counters;
- representative post-AFE board benchmark audio;
- original held-out human qualification WAVs and corpus identity;
- physical target-board raw measurements.

Hosted tests can prove API and evidence contracts. Only the final device/audio path can prove real FAR/FRR, far-field behavior and physical performance. Until those real-world evidence classes and every applicable source-recalibration and tuple-consistency gate pass, `shipping_approved` must remain false.
