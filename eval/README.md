# Continuous-audio evaluation

Release KWS quality is measured on continuous audio, not clip classification accuracy.

The supported flow is:

```text
references.jsonl + WAV corpus
       |
       v
build/kws_wav  (real C runtime, .kwm + .kwk)
       |
       +--> detections.provenance.json
       v
detections.jsonl
       |
       v
eval/score_events.py
       |
       +--> summary.json: FAR/hour, FRR, latency, per-keyword metrics
       |
       +--> false-positives.jsonl
                   |
                   v
          eval/mine_hard_negatives.py
                   |
                   v
          empty-target CTC manifest
```

## Reference format

One JSON object per recording:

```json
{"recording":"living_room_001","path":"negative/living_room_001.wav","duration_s":3600.0,"expected":[]}
{"recording":"positive_001","path":"positive/positive_001.wav","duration_s":12.0,"expected":[{"keyword_id":1,"start_s":3.1,"end_s":4.0}]}
```

Recording IDs must be unique. WAV inputs must be uncompressed mono PCM16 at 16 kHz. `expected` may be empty for negative recordings.

## Audit split isolation first

Before tuning or final qualification, check that training/mining, calibration and final evaluation do not contain the same decoded PCM under different filenames or WAV wrappers:

```bash
python3 training/audit_dataset.py \
  --split train=data/train.tsv \
  --split calibration=data/calibration.tsv \
  --split qualification=data/eval/references.jsonl \
  --audio-root qualification=data/eval \
  --report build/dataset-audit.json
```

The auditor validates mono 16-kHz PCM16 and hashes the decoded PCM payload. It also retains container-file hashes for provenance. A copied/renamed recording or a WAV rewrapped with different RIFF metadata is still treated as the same audio.

## Run the real runtime

```bash
python3 eval/run_corpus.py \
  --runner build/kws_wav \
  --model build/base.kwm \
  --keywords build/xiaowo.kwk \
  --references data/eval/references.jsonl \
  --audio-root data/eval \
  --detections build/detections.jsonl \
  --provenance build/detections.provenance.json
```

`kws_wav` is a hosted file-I/O wrapper around the same C engine used by the product. Heap and filesystem use in this executable do not enter the real-time library. The provenance sidecar binds the exact runner/model/pack/reference/detection bytes.

### Capture a model-bound trace in the same WAV pass

`kws_wav` can optionally retain the exact raw float32 logits and actual
`speech_active`/`end_sample` seen by the decoder, without a second acoustic pass:

```sh
build/kws_wav candidate.kwm candidate.kwk recording.wav recording-id \
  --trace-out new-recording.kwtr --stats-json new-recording.stats.json \
  > recording.detections.jsonl
build/kws_decoder_replay candidate.kwm candidate.kwk new-recording.kwtr recording-id
```

The existing KWTRACE1 header binds the exact loaded model bytes by SHA256,
vocabulary fingerprint, frontend identity and frame geometry. The official
decoder replay retains all its identity checks. No-trace invocation behavior and
event stdout are unchanged. Trace capture exclusively creates its destination;
when combined with `--stats-json`, the stats destination must also be new. Check
the process exit status: an ordinary detected error removes newly created trace
and stats outputs, while stdout may already contain partial events.

Each recording starts from the ordinary empty engine state. EOF is not padded or
flushed; a recording shorter than one 400-sample frame yields a valid empty trace.
Caller chunks remain in the supported 1..320-sample range, and capture checks
sequential frame numbers and the expected final frame count.

Continuous AFE timelines may provide aligned external VAD. KWTRACE1 cannot encode
discontinuity, lost-sample, codec-reopen or clock-reset events: trace mode rejects
any such timeline before PCM processing, including flags on its first span. The
legacy no-trace timeline path still supports them. Trace timestamps are source
input-availability coordinates, not human word-end labels or physical-device
latency measurements. AFE capture-timing fields on live events are not carried by
KWTRACE1; compare canonical event fields when checking replay equivalence.

KWTRACE1 does not bind the keyword pack, effective config, source WAV, AFE
metadata, compiler or executable. Preserve these with the run/build provenance;
do not interpret the trace alone as the complete event-reproduction bundle.
Do not upload private PCM-derived numerical traces by default.

The hosted-build CI step runs `tests/test_wav_trace.py` once per compiler with
one handcrafted model,
three PCM fixtures (0/1/3 complete frames) and two nonempty-trace replays. It checks
actual energy/external-VAD flags, exact logits and model identity, event equality,
EOF, no-clobber outputs, and pre-inference rejection of unsupported timelines.
It runs no training or real corpus and establishes no acoustic-quality result.
This Python subprocess contract is not registered with CTest and does not
qualify the ARM cross-build; it requires native hosted runners.

## Score release metrics

```bash
python3 eval/score_events.py \
  --references data/eval/references.jsonl \
  --detections build/detections.jsonl \
  --summary build/summary.json \
  --false-positives build/false-positives.jsonl \
  --max-far-per-hour 0.2 \
  --max-frr 0.05 \
  --max-p95-latency-ms 500
```

The numeric gates above are examples, not universal product requirements. Define them from the actual use case and acoustic test plan.

### Evidence coverage gates

`run_corpus.py` rejects reference `duration_s` values that differ from decoded
PCM duration by more than one 16-kHz sample. The scorer requires every reference to explicitly
supply `expected` (an empty list means annotated negative audio); omitted
annotations must not silently become negative exposure.

The scorer additionally supports repeatable `--min-expected-per-keyword ID COUNT`,
`--min-negative-hours`, `--min-continuous-negative-seconds`, and
`--max-negative-far-upper-95-per-hour`. FRR gates fail without positive events;
latency gates fail without matched events. Ungated diagnostic summaries retain
legacy zero-valued empty FRR/latency fields, so those values alone are not proof
of quality.

The negative-only upper bound is a one-sided 95% Poisson rate bound using only
recordings explicitly annotated with no expected events. It fails when negative
exposure is absent. Positive audio cannot dilute that denominator. Zero observed
false accepts still gives a positive upper bound (about 2.996/hour for one hour),
so do not substitute a zero observed-FAR target for this confidence-bound target.
This statistical bound assumes representative exposure consistent with a Poisson
count model; repeated identical audio or correlated channels do not create
independent evidence. Run the split/decoded-PCM audit and retain unique source
identities before making an acoustic claim.
A continuous-duration gate requires one sufficiently long negative recording;
many short clips cannot satisfy it by aggregation.

`training/iterate_domain.py` accepts corresponding optional `domain_gates` keys:
`min_expected_per_keyword` (an object mapping canonical uint32 keyword ID strings
to positive integer counts), `min_negative_hours`,
`min_continuous_negative_seconds`, and `max_negative_far_upper_95_per_hour`.
Configured gates apply through `base_gate` to calibration, test candidate
selection and final qualification; missing or non-finite required evidence fails
closed. Existing configurations remain unchanged until a policy is explicitly
set. All domain qualification paths now also require positive `expected` and
`matched` counts to evaluate FRR and latency meaningfully. The far-distance
slice additionally requires a positive integer expected-event count and a finite
FRR in [0, 1]; an absent or empty far slice cannot qualify. Mixed-corpus `far_per_hour` remains a diagnostic and existing gate; neither
it nor synthetic negative-only evidence establishes real-world shipping quality.

The scorer reports:

- total audio hours;
- expected/matched/false-rejected wake events;
- FRR;
- false accepts/hour;
- p50/p95 post-keyword-end latency;
- per-keyword expected/matched/FRR/false-accept counts;
- SHA256 identity for the exact references and detections files.

A detection is matched only to an expected event with the same keyword ID and within the configured pre/post tolerance window. Matching is monotonic and maximizes match count before minimizing total phrase-end timing error, so overlapping windows cannot reuse one detection or cause a simple greedy misassignment.

## Match positive and negative context declarations

`score_events.py --context-manifest context.json --require-matched-context`
rejects incomplete or different positive/negative context declarations before
writing a new score. The check is in the scoring entrypoint, with no dependency
on the research validators. It compares every recording, including recordings
with zero detections, and requires both positive and negative references.

The `eval-context-v1` JSON manifest binds `references_sha256` and
`detections_sha256` to the exact input files. Its `recordings` object must have
exactly the reference recording IDs. Each entry contains `role` (`positive` or
`negative`, checked against the annotations), `audio_sha256` (checked against the
same field in the reference row), and this complete `policy` object:

```json
{
  "sample_rate_hz": 16000,
  "feed_chunk_samples": 4800,
  "appended_context_samples": 4800,
  "appended_context_kind": "digital-zero",
  "reset_frontend": "per-recording",
  "reset_model": "per-recording",
  "reset_decoder": "per-recording",
  "reset_clocks": "per-recording",
  "eof_partial_chunk": "process-retained",
  "eof_flush": false,
  "eof_padding": "none"
}
```

This example declares 300 ms of appended digital zeros; the original short-EOF
condition uses `appended_context_samples: 0` and `appended_context_kind: "none"`.
Appended input and EOF padding/flush are separate fields. Reset fields may also
declare `continuous`; partial-chunk handling may declare `drop`; EOF padding may
declare `zero` or `replicate-right`. Equal complete declarations are required,
not a particular favored policy. Missing/null/`"unknown"` fields never prove
equality. Unsupported fields or values, duplicate JSON keys, input hash changes,
audio identity mismatches and false role declarations are errors.

Without `--require-matched-context`, an explicitly supplied different-condition
manifest remains reportable as `CONTEXT_MISMATCH`; absent or incomplete context
is `CONTEXT_UNVERIFIED`. Existing event scoring and numeric gate behavior remain
available for historical reproduction. The nested `context` result always states
the limited scope; these historical diagnostics must not be described as matched
context evidence. A complete comparison reports `MATCHED_DECLARED_CONTEXT`.

This verifies declarations and file bindings, not the truth of an execution
claim. A self-consistent forged manifest is not attested evidence. Model,
decoder, dataset lineage, representative exposure and source provenance still
need their existing independent checks. `run_corpus.py` does not automatically
emit this manifest or certify unknown external-runner EOF behavior. Neither a
matched context result nor a successful numeric gate grants qualification.

### Saved clips without keyword time alignment

`score_events.py --clip-presence` accepts reference rows with
`expected_keywords: [1]` or `expected_keywords: []` in place of `expected`.
Every row must explicitly declare `annotation_status: "complete"`; missing,
unknown or partial annotations are rejected, never counted as negatives.
It reports positive target presence, negative clips/events, missing keywords,
wrong keywords and repeated events. It does not emit FRR, FAR/hour or latency,
and rejects event metric gates and hard-negative mining output flags. No event
windows are created. Detection `time_s` still must lie in the supplied recording
duration; an input-availability coordinate is not an acoustic word endpoint.
The same context manifest and strict flag apply to this mode.

Prepare the pinned data explicitly before running the offline tests:

```sh
python3 tools/prepare_eval_context_fixtures.py
python3 tests/test_eval.py
```

Preparation downloads only three immutable public numeric logs into ignored
`build/eval-context-fixtures` and verifies exact byte counts and SHA-256 hashes.
To work offline, first use `prepare_eval_context_fixtures.py --source-dir DIR`
with already downloaded, correctly named logs. An optional `--output-dir DIR`
on preparation pairs with `tests/test_eval.py --context-fixtures DIR`.
CI has an explicit preparation step before the scorer tests. Missing, corrupt
or unavailable data fails; tests never fetch data or silently skip regressions.
The source repository retains only the pins, not the raw scientific logs.

`tests/test_eval.py` exercises the actual CLI using the prepared public M1–M5
logs. The original short-EOF condition remains 0/2 positive clips with a
target and 0/3 negative clips with an event; the separately matched 300 ms
condition is 1/2 and 1/3. Combining the 300 ms positives with original negatives
is rejected as a matched claim. These five exposed clips from one Melo voice,
with human words heard on native audio and unreviewed derived representations,
remain diagnostic only. See the fixture provenance in
[`tests/fixtures/eval_context/README.md`](../tests/fixtures/eval_context/README.md).

## Mine hard negatives

```bash
python3 eval/mine_hard_negatives.py \
  --false-positives build/false-positives.jsonl \
  --audio-root data/eval \
  --output-dir build/hard-negative-wav \
  --manifest build/hard-negatives.tsv
```

The generated manifest contains empty CTC targets intentionally. Add it as another `--manifest` to `training/train_ctc.py`, using the **same `--tokens` vocabulary** as the base checkpoint:

```bash
python3 training/train_ctc.py \
  --manifest data/train.tsv \
  --manifest build/hard-negatives.tsv \
  --tokens keywords/tokens.zh.txt \
  --warm-start build/base.pt \
  --head-only \
  --output build/base-hardneg.pt
```

Never mine from the final held-out certification corpus and then reuse the same corpus as unbiased release evidence. Keep at least three distinct pools: training/mining, tuning/calibration and final qualification.

## Recommended corpus buckets

At minimum include:

- clean positives from many speakers;
- near/far field, angle and SPL/SNR variation;
- near-homophones and partial keyword phrases;
- repeated-syllable and repeated-token confusables;
- ordinary conversational Mandarin;
- TV/music/podcast playback;
- local TTS/speaker playback through the shipped AEC path;
- AEC residual and double-talk conditions;
- motor, fan, gear and mechanical noise from the real product;
- silence and low-level room ambience;
- long negative recordings to make FAR/hour statistically meaningful.

All qualification audio must pass through the same BF/AEC/RES/NS/AGC composition and gain policy used by the shipping SKU.

## Observed native-clip readback (no event alignment)

`readback_native_clips.py` consumes a pinned, clean-checkout export from the
`kws-data` repository. That repository owns catalog, source, rights, review and
split validation; this consumer checks receipt identity and actual consumed audio
bytes. It does not parse or normalize the source archive, assign tokens, generate
review receipts, or reinterpret native development roles as fresh test data.

First use the data repository's canonical `python3 -m tools.codex_assets export`
command with exact commit/catalog pins and explicit dataset selections. Save its
JSON receipt outside the clean data checkout and hash it. Then run:

```sh
python3 eval/readback_native_clips.py \
  --receipt "$RECEIPT" --data-root "$DATA_ROOT" \
  --expected-receipt-sha256 "$RECEIPT_SHA" \
  --expected-data-commit "$DATA_COMMIT" \
  --expected-catalog-sha256 "$CATALOG_SHA" \
  --dataset qwen3-xiaowo-reviewed-development-20260928 \
  --dataset qwen3-train-asr-development-20260928 \
  --dataset qwen3-holdout-asr-development-20260928 \
  --runner "$RUNNER" --expected-runner-sha256 "$RUNNER_SHA" \
  --runner-build-receipt "$BUILD_RECEIPT" \
  --model "$MODEL" --expected-model-sha256 "$MODEL_SHA" \
  --keywords "$KEYWORDS" --expected-keywords-sha256 "$KEYWORDS_SHA" \
  --output-root "$NEW_OUTPUT_DIRECTORY"
```

The execution inputs preserve dataset/content identities, native `train`,
`development_a`, `development_b` roles and human/ASR evidence separately.
`execution-inputs.jsonl` deliberately has no `expected` event list, tokens, or word
endpoints. Do not feed it to an event-level scorer. It invokes `run_corpus.py` in
direct C-runner mode, resetting the engine for each unmodified WAV, with no added
silence or decoder overrides. Source WAV identity is checked before execution and
against the runner provenance afterward. Existing output directories are refused.

`clip-readback.json` retains every raw detection, positive target-hit/miss,
wrong-keyword event count, additional same-target events, and confusable event
counts. No detection on a successfully processed positive is a clip miss; a
wrong-keyword event does not count as a target hit. Unknown recording detections
or malformed/out-of-clip events fail closed. An empty detections file is valid
only after all runner invocations completed successfully.

These are observed short-clip development counts, not event-aligned recall,
endpoint latency, continuous FAR/FA-per-hour, fresh holdout results, cross-generator
generalization or product qualification. In particular, target presence anywhere
in a positive clip does not prove a correctly aligned wake event. Frozen model and
keyword-pack hashes, runner identity, build-receipt hash, exporter receipt and
consumer-code hashes remain attached to the report. A manual compiler build must
be identified as such in its build receipt, never as a CMake or full-CI result.

### Saved per-clip event identity contract

`python3 tools/event_readback_contract.py RECEIPT.json` validates the narrow
`kws-event-readback-identity-v1` saved-file contract without running a model,
producer or replay. Its only accepted scope is `observed-per-clip-diagnostic`.
This is an identity and structural consistency check, not a new qualification
or scoring gate. `tests/test_wav_trace.py` exercises artificial positive and
negative receipts on its normal CI path; `--contract-only` runs those fixtures
without invoking C tools.

The receipt binds relative confined paths, byte lengths and SHA-256 digests for
model, keyword pack, original mono 16-kHz PCM16 WAV, nonempty KWTRACE1 trace,
capture/replay binaries, complete supported capture/replay source manifest,
parameter contract, its generated header, event JSONL and completion claim.
The exact fields and supported source closure are defined in
`tools/event_readback_contract.py`. Model/pack vocabulary, trace model identity,
frame coverage, saved VAD flags, finite logits and canonical sample clocks must
agree. Keyword tokens, thresholds, prefix/grace and trailing-blank policy are
read from the bound pack. Runtime settings must match the bound parameter
contract defaults; overrides are rejected.

This first version supports only fresh per-clip engines, zero sample origin,
no padding/flush/discontinuity or extra resets, unchanged PCM, and capture
without a keyword pack (`kws_posterior_dump`). Visible events must identify the
recording/keyword, lie on saved frame clocks, meet the pack threshold and obey
global refractory spacing. Six-decimal printed times/confidences allow
`0.500001e-6` rounding error; two events on the same frame are rejected.
Completion counts must agree with events and consumed frames. These checks
cannot establish that binary execution occurred, that binaries derive from
bound sources, or that logits/events were actually computed by those binaries.

Successful output therefore says `VERIFIED_BINDINGS_ONLY`, with
`execution_observed=false`, `completion_authenticity_verified=false`,
`historical_event_equivalence=false`, and both `production_qualified` and
`promotion_allowed` false. FRR, FAR/hour and word-end latency remain null.
Transcripts alone, extra metric/qualification claims, sealed-test roles,
unknown policies and incomplete identity bindings fail closed. Declared data
roles/label grades are preserved restrictions, not independently authenticated
label evidence. Already observed development clips remain observed data; this
contract does not create event-aligned references or independent test evidence.

### Native decoder replay completion

On native Linux, `kws_decoder_replay MODEL PACK TRACE RECORDING --completion NEW.json`
adds an optional execution completion sidecar without changing event JSONL on
stdout or decoder policy. The destination must not exist (including symlinks).
The tool records engine `processed_frames`, `decoder_hits`, `detections`, and
`refractory_suppressed`; it does not infer internal hits from visible events.

The sidecar uses `kws-decoder-replay-completion-v1`. It binds the loaded model
and pack bytes, the exact private trace snapshot consumed (maximum 64 MiB),
the native executable, compiled parameter-contract digest, actual runtime
settings, recording ID, saved VAD/clock and fresh-engine/reset policy. Invoke
the executable by its filesystem path: completion requires that path's bytes
to match `/proc/self/exe`; emulator/wrapper execution is not supported for this
option. The old CLI remains available without completion.

Publication happens only after complete trace validation and close, stdout
flush, and completion-file flush/fsync/close. A same-directory temporary file
is atomically linked to the new destination without replacing it. Errors or
interruptions before this commit do not create a final success record; partial
stdout or an interrupted temporary file may remain and are not completion.
An interruption after commit cannot undo already completed execution. This is
atomic publication, not a directory-fsync power-loss durability guarantee.

A controlled invocation plus this record supplies actual execution counts.
The saved JSON alone is unsigned and can be forged alongside its bindings;
`event_readback_contract.py` therefore still reports `VERIFIED_BINDINGS_ONLY`,
not execution authentication or production qualification. Its narrow capture
contract still rejects empty captures and non-default runtime overrides even
though standalone decoder replay may successfully process a valid zero-frame
trace or record an explicit override. No FRR/FAR or historical product-event
equivalence is created. Artificial completion tests are included in the
existing `tests/test_wav_trace.py` runner path, separately from its fixed
single-pass acoustic fixture budget.
