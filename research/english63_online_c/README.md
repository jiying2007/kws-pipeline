# Saved English63 FP32 models in standalone streaming C

This is a separate research runtime for the two **already trained** GRU and
DS-TCN candidates from public data run `43595338774`, artifact `14114695344`.
It performs no training, quantization, model selection, threshold adjustment,
label admission, or SDK ABI change. The established SDK's 320-sample hop remains
unchanged. These English candidates require their original 400/160 PCM framing,
32-bin normalized triangular mel filters, Hann window, log floor, and train-only
normalizer. Substituting the SDK frontend would change the experiment.

## Fixed arithmetic and decision contract

`export_weights.py` validates the exact original training-summary SHA and both
saved NPZ SHAs before writing C99 hexadecimal FP32 constants. No pickle or model
object is loaded. Array order, shapes, raw-parameter SHA, original NPZ SHA,
frontend table bytes, and generated-header SHA are retained in `EXPORT.json`.
The C code uses the PyTorch GRU r/z/n gate order, including reset applied to the
recurrent candidate bias; DS-TCN uses causal depthwise convolutions with
dilations 1/2/4 and the existing residual scale 0.5. Both retain the trailing
200-state mean/max readout and original classifier. GRU hidden state has constant
size but an unbounded acoustic past; the finite readout does not truncate it.

Proposal rules stay at 50 observed frames, probability 0.80, three confirming
frames and 100-frame cooldown. EOF adds no samples, frames, scores or events.
The PCM core accepts 1..3199 samples, rejects invalid blocks before consumption,
has an explicit reset, and uses caller-owned fixed state and readonly model
arrays. It has no heap, locks, threads or file access. A synchronous callback is
the boundary to caller-owned output; the CLI does allocation and file I/O.

The radix-2 FFT structure is reused from `src/frontend.c`; FP64 FFT scratch and
FP32 frontend output retain accuracy near low-energy bins. Hann/mel/norm values
are exported from the frozen Python recipe. Dense zero mel entries are omitted
without changing the nonzero FP32 coefficients. Host CPU/native traces determine
the arithmetic result; no ARM/SSC305 speed claim is made in advance.

The criteria below are fixed before the first real C run:

* Maximum absolute normalized-feature, logit and probability errors: **2e-4**
  for each quantity, on every frame; no statistical averaging hides outliers.
* Exact proposal class and real-sample clock agreement between Python and C.
  Proposal probabilities follow the same numeric bound; they are not labels.
* Bit-identical C traces for chunk `160` versus the repeating
  `1,159,161,3199,320,7` plan. Both plans reset their chunk index at source-file
  boundaries, while continuous-model state and sample clock never reset.
* Original saved proposal class/sample identities must also be reproduced by
  the Python reference. Original runner bitwise probability hashes are recorded
  as diagnostics because CPU math dispatch can differ across runner machines;
  this does not waive the frozen event identities or C/reference error bounds.

## Public CPU runner execution

Install the already used Python 3.12.16, NumPy 2.3.5 and Torch 2.10.0+cpu runtime
with GCC. The input trained-result directory is the unmodified successful
artifact root, containing `model/SUMMARY.json`, `model/events.jsonl` and both NPZs.

```sh
python -I -B research/english63_online_c/c_replay.py \
  --tts-zip /absolute/original-tts.zip \
  --review-json /absolute/original-joined-review.json \
  --trained-result-dir /absolute/original-successful-result \
  --output-dir /absolute/new-comparison-directory
```

The source-bound existing loader/auditor verifies all 63 original recordings,
their 2,101,760 actual PCM samples, source identities and 30/12/21 roles. No role
enters an optimizer. Python computes each model once per isolated clip and once
on the full fixed-order concatenation. C runs both chunk plans in both modes.
Temporary PCM is byte-checked against the original decoded PCM hash and removed.
The outputs aggregate traces rather than retaining per-clip files or audio.

The native CLI can be embedded separately after array conversion and compilation:

```sh
en63_replay --supervise gru continuous 160 /absolute/pcm-paths.txt /absolute/new.trace
en63_replay --supervise dsc_tcn continuous mixed /absolute/pcm-paths.txt /absolute/new.trace
```

The manifest has one original PCM16LE path per line. `isolated` resets at each
file; `continuous` does not. Trace records contain a stream index, actual sample
clock, 32 features, three logits, three probabilities and proposal class. All
binary fields are explicitly little-endian and the record is 168 bytes.
Use `-` instead of the trace path for native profiling without trace-output I/O.

## Opt-in pooling traversal

`EN63_POOL_TIME_MAJOR=0` is the default and retains the original channel-first
pooling loop. Compile the same source with `-DEN63_POOL_TIME_MAJOR=1` to visit
the ring in its original chronological order and update all 32 channels in
each contiguous row. Each channel keeps the original FP32 addition order and
strict `>` maximum comparison, including the first equal signed-zero maximum.
The index formula is unchanged and is evaluated once per row. Both branches
use the existing `pooled[64]` scratch; caller state, API, ring storage, weights,
division, classifier and event decoder remain the same. Values other than
0 or 1 and fast-math/finite-only compilation are rejected. The paired builds
use `-ffp-contract=off -fno-fast-math`.

The source fixture compares the actual two compiled pooling implementations
for all 200 nonempty counts and all 200 ring write positions, using five
invented finite-value patterns: signed zeros, alternating `FLT_MAX`, ordered
cancellation, normal/subnormal minima, and mixed signs/exponents. These are
200,000 cases and 12,800,000 FP32 output comparisons. Separate invented,
nonzero GRU/DS-TCN tensors and generated PCM compare complete 401-frame traces,
events and clocks across both variants, 160/mixed chunks, reset and gap recovery.
No saved project header, checkpoint or audio is loaded by these fixtures.

```sh
python3 -I -S -B research/english63_online_c/pool_contract_cases.py --cc gcc
python3 -I -S -B research/english63_online_c/pool_contract_cases.py --cc clang
python3 -I -S -B research/english63_online_c/pool_contract_cases.py --cc gcc --sanitize
```

`check_source.py` also invokes these fixtures. Their success establishes the
source contract. CPU benefit requires the separately bounded, paired public
Actions measurement with the original fixed inputs and saved byte-identity
evidence; it is not inferred from source-level index-operation counts.

## Resource accounting and saved evidence

Each native launch uses a fresh child of a small native supervisor, created
before state allocation. This prevents inherited high-water RSS of the Python
or Torch parent from being mislabeled as native C RSS. The child reports its
own Linux `RUSAGE_SELF` high-water RSS through inference completion, CPU/wall
times, raw feed durations and quantiles. The RSS observation occurs before the
CLI copies/sorts timing samples and writes JSON, so it is not a full-process-
lifetime peak measurement. The dedicated continuous/160 profile disables trace I/O. It still includes the
CLI's PCM input file reads in process CPU/wall time; feed timings wrap the core
call only. State bytes, FP32 parameter bytes, frontend-table bytes, binary bytes
and process RSS are distinct. Stack usage is **not measured**. QEMU, if used,
is correctness evidence only, never a performance/physical-SSC305 measurement.

Successful output has 17 files: the C header/export manifest, native executable,
build provenance, two compressed Python reference traces, eight native traces,
resource observations, `SUMMARY.json` and `REPLAY.json`. Original WAVs/NPZs are
not duplicated. The verification receipt hashes the complete output tree.

```sh
python3 -I -S -B research/english63_online_c/c_replay.py \
  --verify-result-dir /absolute/saved-comparison-directory
```

This final command uses only stdlib, verifies saved hashes and strict measured
summary bounds, and performs zero model forwards. It is artifact verification,
not a claim of another numerical run. Partial failures remain failed artifacts;
do not change a threshold or tolerance to turn a disagreement into success.

Local verification is source/analytic-fixture only:

```sh
python3 -I -S -B research/english63_online_c/check_source.py --cc gcc
python3 -I -S -B research/english63_online_c/check_source.py --cc clang --sanitize
```

The fixtures use invented zero/analytic tensors, not either project checkpoint.
Real audio/model execution belongs exclusively to the authorized public runner.
All receipts keep formal training admission, human gold, promotion and selection
false; independent FRR, continuous-negative FAR/hour, word-end latency and
SSC305 performance remain `NOT_RUN` even if the C implementation agrees.

The first real saved-weight comparison completed successfully in public run
`43605922752`. See the [measured result and its limits](../../docs/KWS_V4_ENGLISH_NATIVE_C_20261114.md)
and [machine-readable receipt](measured-20261114.json).

## Explicit discontinuities: a separate bounded state experiment

`en63_reset` starts a new session and clears all counters. The added
`en63_discontinuity` clears PCM overlap, encoder history, the trailing pool,
confirmation/cooldown and the 50-observed-frame warmup while preserving the
session's supplied-PCM sample count, cumulative processed frames and proposal
count. `segment_samples` and `segment_frames` describe the current uninterrupted
segment; `discontinuities` counts accepted notifications. Neither operation
creates a frame or callback. Discontinuity rejects finished/faulted states and
all reentrant calls. A finished or faulted state can start a new session through
explicit reset. The product KWSP ABI is unchanged; research callers must compile
with the matching header and allocate `en63_state_bytes()` bytes.

After a gap the callback reports its actual cumulative supplied-PCM sample
index, rather than reconstructing that index from a frame number. Previously
supplied, unfinished tail samples remain counted even though their acoustic
carry is discarded. Missing capture samples, wall time and acoustic word ends
are not inferred. An ordinary file boundary is not a discontinuity unless the
caller explicitly reports one.

The new public Actions entry is stdlib-only apart from GCC:

```sh
python3 -I -S -B research/english63_online_c/state_replay.py \
  --baseline-result-dir /absolute/unpacked-native-c-43605922752 \
  --tts-zip /absolute/original-english63.zip \
  --review-json /absolute/original-joined-review.json \
  --output-dir /absolute/new-state-contracts

python3 -I -S -B research/english63_online_c/state_replay.py \
  --verify-result-dir /absolute/saved-state-contracts
```

The first command requires first-attempt public Actions execution metadata. It
uses the already exported FP32 header from artifact `14117638750`; it neither
exports weights again nor imports Torch or NumPy. All 63 original WAV/PCM
identities and the 30/12/21 source roles are checked, but only the first **128,246
original samples** of the fixed sorted-cell concatenation enter the experiment.
Two 64,123-sample windows provide distinct inputs. This is an exposed diagnostic
selection, not new independent audio or a new measurement on all 63 recordings.

Each model has ten fixed cases: old saved baseline prefix, midstream reset,
reset after EOF, two interleaved instances, discontinuities at samples 399,
400 and 32,401, unpadded EOF, invalid-call atomicity and invalid-model atomicity.
The first cut precedes a complete frame; the second follows one frame; the third
follows 201 frames and leaves a non-hop-aligned tail after a full readout pool.
These are programmatic gap notifications between saved PCM bytes, not measured
capture overruns. Each model completes exactly 4,773 native frame forwards and
consumes 769,475 samples including repeated diagnostic reads. Repeated inputs
contribute **zero new independent audio samples or seconds**.

The unchanged baseline must reproduce all FP32 bytes, proposal classes and
sample clocks of the old continuous trace's first 399 frames. Original `2e-4`
error bounds are retained as diagnostics and do not waive this new bit-identity
gate. Reset/interleaving must match fresh controls. Every post-gap suffix must
match a fresh suffix at every FP32 value and proposal, with the preserved
session clock offset. Fixed thresholds and checkpoint identities are unchanged.
No failure authorizes a retry or a relaxed comparison.

Seven case results are recomputed from saved numeric traces. The three EOF and
invalid-input/model cases additionally rely on the executed C return-code and
full-state `memcmp` guards, fixed guard counters, native exit status and their
original stdout. Their evidence is labeled `executed_native_guard_receipt`;
full state snapshots are not retained for independent offline reconstruction.
The saved verifier re-parses both native stdout records and binds them to the
parsed observations. The trace proves frame/proposal clocks and contains no
padding or EOF-only observations.

A successful result has exactly 16 files: two complete state traces, two native
JSON records, two stderr logs, two unchanged baseline-prefix traces, exported
header, executable, `BUILD.json`, `INPUTS.json`, `PROTOCOL.json`, `SOURCE.zip`,
`SUMMARY.json` and `REPLAY.json`. The last receipt covers the other 15 files;
missing or extra files are rejected. Failure retains its partial files and
`FAILED.json`, without a successful verification receipt. Raw PCM is temporary.

The native supervisor also records Linux `wait4` peak RSS after child exit.
That is a process-lifetime observation including this test harness, input
buffer, trace I/O and output serialization. It is separate from the old
inference-end RSS observation and is not a core-only, latency, real-time or
SSC305 benchmark. Overflow and fault gates use explicit test-state fixtures;
they are not a 24-hour soak. The two finite-window weak-MIL checkpoints remain
research proposals, with acoustic truth, training admission, selection, natural
negative FAR/FRR and word-end latency unqualified.

`state_contract_cases.py` tests only invented saved binary records. Existing
`check_source.py --cc gcc` / `--cc clang` also compile and run the new driver with
their own invented constant-classifier header and generated PCM; no original
checkpoint or original audio enters those analytic fixtures.
