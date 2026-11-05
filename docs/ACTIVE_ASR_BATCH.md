# Active local ASR batch

This entry runs the existing Qwen ASR 0.6B, SenseVoiceSmall and native Whisper
small checkpoints on existing audio. It creates no TTS audio, uses no paid model
API, and publishes nothing. The old frozen campaign scripts and locks remain
unchanged. The source and output files belong in a private directory outside the
Git worktree.

## Completed D0 execution

On 2026-11-05, CPython 3.12.16 and the combined ASR environment completed all
64 clips for each of the three models: 192 native calls, 192 saved results,
zero repeated model/clip calls, and no model-process failures. The full stage
reused its six smoke results and completed in 457.06 seconds; elapsed time from
the original job creation to completion was 694.64 seconds. All 727 inventoried
input/output files passed a subsequent hash readback.

The original source declares four synthetic voice presets and `train_only` for
all clips. Recognition output was nonempty and terminated normally in every
cell; those execution checks do not establish human transcription accuracy,
acoustic completeness or KWS quality. Whisper's Chinese language condition
remains disclosed. Audio, native transcripts and the complete result package
are retained privately; they are not included in the public repository.

The final summary SHA256 is
`8c65e59d2da86a04163dbb650290d0b624038eaa162da316605609c447022570`;
the 727-file inventory SHA256 is
`ab84afceb8531e0e98eb503ecb836ff1ed7058e4d7d4614bd6e161cfdc611ce3`.

## Prepare the environment

Use the active interpreter from `configs/runtime/research-runtime.json`. If it
comes from an extracted Linux Actions archive, retain its `lib` directory in
`LD_LIBRARY_PATH` for the base interpreter and its venvs; see
[`RESEARCH_RUNTIME.md`](RESEARCH_RUNTIME.md).

```sh
python3.12 tools/setup_asr_runtime.py --root /private/runtime-asr
```

The helper verifies and combines the two retained ASR dependency inventories.
The 125 unique package artifacts are acquired by exact URL and SHA256. Duplicate
names must agree in version, filename and body hash. It installs the wheels with
dependency resolution disabled, builds the five retained source distributions
using the pinned build tools, and runs `pip check` plus the active role check.
Whisper source is acquired file by file from its retained source lock. All model
assets are verified by their retained hashes before use.

Setup has a bounded wall-clock budget, one network attempt per artifact per
invocation, and at most four download workers. Valid cached artifacts are reused.
A failed invocation retains its own receipt; another invocation creates a new
attempt and reuses only verified files. It never declares model inference from
an installation or import check.

## Bind audio and run

```sh
/private/runtime-asr/venv/bin/python tools/active_asr_batch.py prepare \
  --archive /private/existing-audio.zip --source-sha256 EXACT_SHA256 \
  --expected-count 64 --output /private/runtime-asr/inputs

/private/runtime-asr/venv/bin/python tools/active_asr_batch.py run \
  --runtime /private/runtime-asr --inputs /private/runtime-asr/inputs/inputs.json \
  --output /private/runtime-asr/results --phase smoke --attempt smoke-001 \
  --threads 4 --budget-minutes 90

/private/runtime-asr/venv/bin/python tools/active_asr_batch.py run \
  --runtime /private/runtime-asr --inputs /private/runtime-asr/inputs/inputs.json \
  --output /private/runtime-asr/results --phase full --attempt full-001 --threads 4
```

The original mono FLOAT 24 kHz WAV bytes are retained. A separately identified
16 kHz float32 derivative uses `scipy.signal.resample_poly` with up/down 2/3,
Kaiser 5.0 and constant padding. There is no normalization, clipping, trimming,
or PCM16 quantization. Both original WAV and original/derived sample hashes are
recorded. Non-audio members of the source ZIP are not read as input context.

The smoke phase runs the first and last opaque clip IDs for each model. The full
phase requires those six completed execution records and reuses them. A single
process owns each model, models execute sequentially, and a runtime-wide file
lock prevents concurrent heavy model workers. Torch and numerical thread pools
are capped at four. The persisted job deadline applies across attempts; resuming
does not reset the clock. The maximum accepted wall-clock budget is 999 minutes,
with a 90-minute default. A 64-clip job permits at most 192 durable call claims.

Claims are written before recognition. Completed identical results are reused;
a completed cache entry must also retain its native output with the recorded
SHA256 and matching schema, model, clip and recognition identity. Missing or
changed native evidence is rejected without another recognition call.
A claim without a completed result remains consumed and is not automatically
recognized again. An exception stops that model process and retains the attempt
receipt. Quality problems are saved as evidence rather than retried. A repaired
implementation uses a new attempt ID. The controller can resume selected models
with `--models`, while retaining the same input and job identity.

Qwen uses native automatic language selection with an empty context and greedy
generation. SenseVoice uses automatic language selection, no VAD or ITN, native
fbank and zero dither. Whisper uses native greedy short-form decoding with
**Chinese (`zh`) supplied explicitly**, no prompt, prefix or timestamp output.
These are independent transcript attempts under disclosed model conditions;
the three-way result is not independent discovery of the language. No intended
phrase or peer transcript reaches any recognizer.

## Output contract for weak-label consumers

`inputs.json` maps `clip-NNN` to `original_member` and the original/derived audio
identities. Join on the original WAV hash or that mapping, not transcript text.

| Location | Schema | Contents |
|---|---|---|
| `job.json` | `active-asr-job-v1` | Input identity, model list, total call cap, persisted deadline, evidence boundary |
| `claims/<model>/<clip_id>.json` | `active-asr-call-claim-v1` | Durable before-call claim, exact input/model identity, attempt |
| `native/<model>/<clip_id>.json` | `active-asr-native-output-v1` | Original native text/tokens/metadata and decode trace where available |
| `results/<model>/<clip_id>.json` | `active-asr-result-v1` | Consumer projection linked to the native file by SHA256 |
| `attempts/<attempt>/<model>.json` | `active-asr-worker-v1` | Stage, reuse count, new call count, error or completion, resource observations |
| `attempts/<attempt>/<model>-load.json` | Model-specific | Verified load/architecture evidence |
| `attempts/<attempt>/summary.json` | `active-asr-attempt-v1` | Child exit status, saved counts and elapsed time |

Model keys are `qwen06`, `sensevoice`, and `whisper`. Each completed result has:

- `identity_sha256`, `model`, `clip_id`, `attempt` and `status`;
- `input`, including `original_wav.sha256` and `derived_pcm.sha256`;
- `model_identity`, including model revision/assets, runtime lock and installed
  package identities, Python version, native decoder configuration and threads;
- `text`, `quality_flags`, `decoder_terminated`, and `transcript_extraction`;
- `native_output_sha256`, `decoder_source_sha256`, elapsed time and completion time;
- `boundary`, explicitly marking human gold, training admission and shipping
  approval false, and acoustic verbatim completeness unknown.

`status: complete` means that the native call returned and its output was saved.
It is not an accuracy verdict. Consumers must inspect quality flags and native
termination and must preserve disagreement, empty text and incomplete outputs.
Qwen text extraction removes its native language/ASR prefix; SenseVoice removes
only control tags; Whisper retains `DecodingResult.text`. Raw outputs are retained
separately. No transcript repair, spelling normalization or intended-text
substitution is applied.

The model workers use local assets, offline library configuration and a Python
socket audit guard. This is not a claim of kernel network isolation. Unit tests
exercise artifact identity, ZIP boundaries and durable reuse semantics without
downloading dependencies or constructing models:

```sh
python3.12 -B tests/test_active_asr_batch.py
```

After the controller exits, retain a full matrix audit and content inventory:

```sh
python3.12 tools/summarize_active_asr_batch.py \
  --runtime /private/runtime-asr --inputs /private/runtime-asr/inputs/inputs.json \
  --output /private/runtime-asr/results
```

The auditor checks every original WAV and derived PCM identity, all model/clip
pairs, claims, native bodies, consumer projections and runtime/configuration
bindings. It records all input/native/result hashes, worker error/quality flags,
small-sample reuse, elapsed times and the measured per-process peak RSS. It also
checks the declared input context and native call sites; this is not an OS
file-access trace. Raw exact-string agreement is reported separately and is not
an accuracy measure or a normalization policy.
