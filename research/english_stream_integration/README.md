# English requested-intent research streaming component

This is a source preparation patch. No local C compilation or execution, real
model inference, acoustic evaluation, or installation has been performed by its
source tests. `integration_pins.json` is disabled and contains no invented future
head/fullchain run identities. The separate fullchain numerical worker remains
unchanged. A future explicitly frozen public Actions job must authenticate the
actual successful fullchain artifact before enabling this integration study.

## Scope and source boundary

`ekws_research_stream.c` composes the existing verified mel frontend and saved
causal head. CMake checks the original mel implementation/header, all eight head
source files, and all three generated files by SHA-256. It neither re-exports a
graph nor changes kernels, normalization, weights, history, threshold (0.80),
three-confirmation rule, or 13-hop cooldown. A new pipeline commit may add this
wrapper; the entire checkout is not required to retain the old `6ed95…` HEAD.

The public namespace is **KwsEnglishResearch::stream** and the model format is
**EHKWS01**. It has no KWSP/KWKP compatibility or effect on `KwsPipeline::core`.
The actual two keywords are class 1 **Hey Vosen**, class 2 **Hey Mero**, with class
0 background intent. Outputs remain uncalibrated weak-research proposals. Human
gold, pronunciation verification, formal training admission and promotion stay
false; independently measured FAR/FRR, word-end latency and SSC305 remain separate.

## Public interface and ownership

Include only `ekws_research_stream.h`. Query `ekws_stream_bytes()` and alignment,
then provide a caller-owned inactive arena and a separately aligned float
workspace of `ekws_workspace_bytes()`. `ekws_stream_init()` loads the exact-sized
197,908-byte EHKWS01 blob into the arena after its caller has authenticated the
saved artifact/weight hashes. It checks format and finite scale/weights; it is not
a cryptographic artifact-verification service. The blob can be released after
successful initialization. Failed initialization leaves the output pointer
unchanged but invalidates the candidate arena. Keep the previous active stream
and its storage separate until the new initialization succeeds.

The arena includes the copied head, its history, the frontend ring and counters.
The verified graph workspace is 528,384 bytes. Use the query functions for actual
native/ARM arena size; do not equate arena plus workspace with total RSS/PSS or
include the head twice. Generated constants, code, I/O buffers, loader and OS
memory are additional. The hosted runner owns its own allocations and file I/O;
the core makes no allocation, I/O, thread or resampler calls.

Call `ekws_stream_accept_pcm16(stream, pcm, count, first_sample, &consumed)` with
mono PCM16 already at 16 kHz. `first_sample` is the zero-based count of observed
input and must equal `stats.next_input_sample`. Arbitrary chunk sizes are legal,
and one call may deliver multiple callbacks. `consumed` is authoritative on
partial failure. Zero input emits nothing; there is no EOF flush or silence pad.
Input/output/config pointers must not alias the arena, workspace or PCM storage.

The event callback receives a borrowed `ekws_event` with keyword ID, original
head probability, and `available_after_sample`. Copy it to a bounded queue if it
must outlive the callback. This clock is when the evidence became available; it
is not a measured word-end label or hardware wall-clock callback time. The
optional score observer receives the existing ready flag, event, three logits
and three probabilities after every actual embedding, including the startup
non-ready outputs. It performs no additional head/model evaluation. Normal
applications leave the score observer unset.

One stream has one owner. There is no internal locking. Every API rejects
reentry during feed/callback execution, including stats snapshots. Publish
events across threads; do not mutate or inspect the same stream concurrently.
Callbacks must return zero on success. Failure poisons the stream, preserves
actual consumption and counts, and forbids further feed until reset or explicit
discontinuity. Already consumed PCM and delivered events must not be retried.

`reset()` clears frontend and head state, starts a new logical recording at
sample 0, and retains saturating lifetime counters. `discontinuity(next, reason)`
clears both histories and cooldown while preserving an explicitly declared
absolute timeline; `next` cannot move backwards. A device clock reset therefore
needs either a maintained monotonic application offset or a new logical reset.
No skipped samples are invented. The first embedding after a boundary requires
16,000 real samples; the first ready head requires 16 embeddings (35,200 samples
after that boundary). Snapshot bounded statistics on the owner/control path.

## Default-off build and install

`main-cmake-opt-in.patch` adds only the default-OFF main option and one guarded
`add_subdirectory(research/english_stream_integration)`. Copy this directory into
that research path when a protected source PR is prepared; keep the patch as a
review artifact rather than applying it to any consumed numerical profile.

Future authorized Actions can build the subdirectory independently:

```sh
cmake -S research/english_stream_integration -B BUILD \
  -DKWS_ENABLE_ENGLISH_RESEARCH_STREAM=ON \
  -DEKWS_PIPELINE_SOURCE=PIPELINE_SOURCE \
  -DEKWS_GENERATED_SOURCE_DIR=VERIFIED_FULLCHAIN/generated \
  -DCMAKE_INSTALL_PREFIX=INSTALL_PREFIX
cmake --build BUILD --parallel 2
cmake --install BUILD
```

The root opt-in uses the same options and does not alter the old core. ARM uses
the declared `arm-linux-gnueabihf-gcc` toolchain, a Linux cross-system setting,
static runner linkage and the same no-FMA/no-reassociation flags. Compiler
configure/build/install work belongs to one bounded build session per backend.
There is no `try_run`, CTest replay, benchmark, or automatic inference at configure,
build or install. Do not execute these commands locally in this task.

The installation contains `lib/libekws_research_stream.a`,
`include/ekws_research_stream.h`, `bin/ekws_pcm_stream`, and the relocatable
`lib/cmake/KwsEnglishResearch` package. Both original licenses are hash checked
and installed under `share/KwsEnglishResearch/licenses`; the upstream license
defaults to `VERIFIED_FULLCHAIN/licenses/openWakeWord-LICENSE`. Consumers use:

```cmake
find_package(KwsEnglishResearch 0.1.0 EXACT CONFIG REQUIRED)
target_link_libraries(application PRIVATE KwsEnglishResearch::stream)
```

Record actual install leaf paths (GNUInstallDirs may be configured), compare
installed header/library/runner bytes with the reviewed sources and build
outputs, and invoke the **installed** runner for the one integration process.
That folds actual installed-binary/API use into the fixed process budget.
The CMake export is read back as source; no second consumer inference is needed.

## Normal stdin runner

```sh
INSTALL_PREFIX/bin/ekws_pcm_stream VERIFIED_HEAD.bin recording_id < audio.pcm
```

Recording IDs use 1–128 ASCII letters, digits, `_`, `-`, or `.`. The runner reads
320-sample blocks, rejects an odd final byte, and writes only event JSONL to
stdout. Each row contains `recording`, `keyword_id`, recording-relative `time_s`,
`confidence`, `available_after_sample`, `input_origin_sample` and explicit
research-only flags. It never appends PCM at EOF. Human/independent scoring needs
separately authenticated audio, labels, rights and AFE timing; stdin availability
alone does not establish live microphone integration.

## Fixed integration experiment: two builds, two processes

The future job has one native and one ARM/QEMU installed-runner process:

```sh
INSTALL_PREFIX/bin/ekws_pcm_stream --verify-cached VERIFIED_HEAD.bin \
  zero/input.pcm zero/NATIVE_OR_ARM.bin \
  sine400/input.pcm sine400/NATIVE_OR_ARM.bin \
  lcg14/input.pcm lcg14/NATIVE_OR_ARM.bin
```

All six input paths and the head come from one authenticated successful fullchain
artifact. No old binaries are run and no new reference is computed. Each process
consumes each 48,000-sample PCM exactly once with chunks `1,1279,257,4096`:

1. zero: reset, input origin 0;
2. sine400: reset, input origin 0;
3. lcg14: explicit route discontinuity after sample 48,000, input origin 48,000.

For the third case, saved relative trace clocks map to absolute clocks by the
predeclared +48,000 offset. No audio gap filler is added. Every actual head output
(26 per case, including 11 ready outputs) must match the same-backend saved
stage4 logits/probabilities by float32 bits, and ready/event/clock must match
exactly under that mapping. Event callbacks are checked independently too.
An empty event list cannot bypass the 78 score observations. The process also
checks empty calls, rejected wrong-clock input, both cleared histories, rejection
of callback reentry, and cumulative accounting, with no extra model window.

PCM and reference files are read before timing. Each accept call is surrounded
by `CLOCK_MONOTONIC`; times are accumulated per case, including in-memory score
comparison callbacks and excluding PCM/trace loading and report I/O. JSON is
written only after processing. This is host integration profiling, without a
speed pass threshold; it is not the old trace-writing process scope, a matched
benchmark, a physical board measurement, or a word-end latency observation.
The report saves all observed head bit patterns and any completed partial cases.

The disabled plan permits two build sessions and exactly two inference processes,
each at most 144,000 samples / 78 embeddings / 33 ready outputs. There are zero
new ORT calls, optimizer updates, TTS/ASR/phones or real-audio cases. A failure is
retained without reattempt. Missing process/terminal receipts remain unknown.
Actual once acquisition/guard/upload orchestration is a later **kws-data** public
main first-successor Actions integration, checking out one pinned **kws-pipeline**
component commit. It is not an enabled executor supplied by this source patch.

## Source checks

`python3 -I -S -B test_stream_source.py` uses only stdlib and invented traces and
checks exact core hashes, the source/interface/plan contracts and main CMake
patch applicability. Run also with `-O` and `-OO`. It does not compile, run C,
load actual weights, execute a model, or assert that this new component is yet
numerically or operationally verified. Those checks belong to the bounded
public Actions experiment after genuine fullchain evidence exists.

The separate `english-research-stream-contract.yml` is a public PR source-only
workflow patch. It runs those three Python modes and GCC `-fsyntax-only` against
the fixed core headers plus a temporary generated header containing only two
extern graph declarations. It produces no object, linked program or model
execution. Its inline check confirms the root option is OFF and guarded; the
existing root CI continues to cover default-off CMake builds without duplication.
This workflow is not run locally by the preparation task.
