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

## Resource accounting and saved evidence

Each native launch uses a fresh child of a small native supervisor, created
before state allocation. This prevents inherited high-water RSS of the Python
or Torch parent from being mislabeled as native C RSS. The child reports its
own Linux `RUSAGE_SELF` peak RSS, CPU/wall times, raw feed durations and quantiles.
The dedicated continuous/160 profile disables trace I/O. It still includes the
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
