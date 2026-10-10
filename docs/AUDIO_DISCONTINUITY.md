# Audio discontinuity handling

A streaming KWS engine must not bridge missing or unrelated audio. Product integrations should notify the engine whenever capture continuity is broken by XRUN, route change, clock reset, device suspend/resume or equivalent pipeline reset.

The discontinuity operation clears partial frontend samples, PCEN smoothing state, recurrent hidden state, decoder/pending prefix state and the active refractory window while preserving configured keywords, processed counters and cumulative telemetry. A discontinuity counter is retained for soak diagnostics.

Calling the discontinuity API from the same serialized owner context as `kws_engine_accept_pcm16()` preserves the single-owner runtime contract.

## Opt-in sample-aligned VAD and replay

The existing `KWS_FRAME_METADATA_API_VERSION` remains **1**. Its external VAD
probability belongs to the block that completes a feature frame, preserving
existing caller behavior. `kws_engine_accept_pcm16()` without metadata is also
unchanged. New adapters may explicitly select
`KWS_FRAME_METADATA_ALIGNED_API_VERSION` (**2**, same struct layout):

- A metadata probability describes every sample in that call. Split calls at
  VAD boundaries, even when the transport delivers larger or irregular chunks.
- VAD v2 uses the sample-weighted mean over the complete 400-sample feature
  window, including the 80-sample overlap. The existing threshold is unchanged.
- If any sample in that window lacks valid v2 VAD, use the existing frame-energy
  gate. Discontinuities/reset clear both frontend overlap and VAD history.
- No heap, filesystem, locks or new tunable parameters enter the runtime. The
  bounded VAD ring increases the measured 64-bit hosted arena from 22,168 to
  23,792 bytes (+1,624); integrations must use `kws_engine_required_bytes()`.

Selecting v2 changes external-VAD semantics and requires recalibration before
shipping qualification. It does not establish an acoustic improvement. No model,
decoder, keyword threshold, parameter-contract default or config hash changes.
The public additive API should accompany the next software/package version bump.

`kws_wav ... --metadata-tsv recording.timeline.tsv [--block-samples 160]`
replays this explicit v2 path. Blocks may be 1..320 samples; the default remains
160. The host adapter validates the entire sidecar before producing detections,
then splits at both delivery and metadata-span boundaries. It does not invent
VAD for a plain WAV. `eval/run_corpus.py` passes `metadata_path` from a reference
(relative to `--audio-root`), verifies `metadata_sha256` before/after execution,
and binds that sidecar in provenance. The existing posterior cache rejects
metadata input because it does not retain VAD/timeline semantics.

The UTF-8/ASCII TSV starts with exactly `kws-afe-timeline-v1` and a newline. Each
following line has nine whitespace-separated fields, with no optional columns:

1. Accepted output start sample, beginning at zero
2. Positive sample count, at most uint32; spans exactly cover the WAV
3. Stream sequence, consecutive unless an explicit reset flag is set
4. Capture-clock availability timestamp of this span's first output sample, ns
5. Existing metadata flags (1=discontinuity, 2=XRUN, 4=codec reopen,
   8=clock reset, 16=external VAD valid; flags may be combined)
6. Lost input samples before this span; nonzero requires a reset flag
7. VAD probability in [0,1]; ignored when flag 16 is absent
8. AFE delay in samples, uint32
9. Lowercase 64-hex AFE config SHA-256

Without clock reset, the next timestamp must equal the previous timestamp plus
(previous span count + newly lost samples) × 62,500 ns. Delay/config changes
require a reset flag. Clock reset explicitly begins a new capture epoch and may
restart timestamps/sequence. Unexplained timestamp jumps, sequence gaps,
configuration changes, noncanonical fields or incomplete coverage are errors.
The SDK metadata API itself still treats sequence/time/config as telemetry;
these stronger continuity checks belong to the opt-in host adapter.

`end_sample`/`time_s` keep counting accepted output PCM, excluding missing
samples. They are never silently relabeled as raw capture time. Sidecar-mode
detections additionally report:

- `output_end_sample`: existing exclusive output boundary
- `capture_epoch`: increases on declared clock resets
- `decision_capture_ns`: capture-clock sample availability at that boundary
- `raw_acoustic_end_ns`: that coordinate minus the declared AFE delay

The last value maps the inference frame boundary, not an aligned word end. It
is signed (leading delay padding can map before zero). These are
sample-coordinate mappings, not measured wall-clock callback/inference latency.
Do not subtract across clock epochs. A gap preserves the accepted-output counter
but resets frontend/recurrent/decoder/VAD state; lost samples remain separately
visible in stats and the sidecar mapping.

## Constant-delay final-corpus timing

An AFE result sidecar can opt into `timing_contract: "afe-output-sample-v1"`.
It declares a constant sample delay at 16 kHz, zero-origin continuous output,
and no capture gaps/config changes. `latency_samples` remains required. Optional
`vad_segments` cover every output sample using `start_sample`, `sample_count`,
and `probability`; omitting a segment's probability requests energy fallback.
`run_final_afe_corpus.py` emits the hash-bound replay TSV, raw reference start/end
coordinates and shifted output references. It rejects discontinuity declarations
rather than applying a scalar shift to a piecewise timeline. Direct TSV replay
above supports discontinuities for integration tests; discontinuous final-corpus
scoring needs independently annotated piecewise mappings and is not claimed here.

Legacy result sidecars without this declaration retain the existing output
references and energy-only replay. Unmarked historical results are not rewritten.
The scorer retains all old matching and clipped post-end latency gates, adding:

- `p50/p95_signed_end_offset_ms` and `early_detection_count` for matched events
- Fully declared corpora: `p50/p95_post_afe_signed_end_offset_ms` (decision minus
  shifted output-word end) and `p50/p95_raw_capture_signed_end_offset_ms`
  (decision minus original raw-word end, including AFE delay)
- `timing_coordinates` coverage/status; mixed or unmarked inputs suppress the
  explicit raw/post-AFE aggregates. No matches produce null explicit metrics.

Raw/output reference mappings are validated. These additive diagnostics do not
replace the historical gate or prove wall-clock end-to-end product latency.
Synthetic unit tests cover chunk equivalence, missing VAD, gap/config/clock
handling, malformed timelines and nonzero AFE delay. They are not real-audio,
physical-board, threshold-calibration or shipping evidence.
