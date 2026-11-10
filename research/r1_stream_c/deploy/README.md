# Strong R1 FSMN: standalone reference64 research SDK

This opt-in package keeps **R1-DEV-FSMN-20261107** weights, vocabulary, frontend
assets and the separately validated reference64 arithmetic. It does not change
the default runtime or repair/relabel the historical FP32-vs-Torch failure.
The output is an **uncalibrated full-keyword proposal**, not a qualified wake
command. No optimizer, TTS, ASR, threshold search or data admission is involved.

## Generate, build and run

From a complete pinned checkout, with the owner's original unpacked model
package (its original MODEL.json and all bound members, not a guessed checkpoint):

```sh
python3 -B research/r1_stream_c/deploy/materialize.py \
  --repo . --model-package /owner/KWS_R1_DEV_FSMN_20261107 \
  --output /new/r1-sdk
cmake -S /new/r1-sdk -B /new/r1-sdk-build \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/new/r1-sdk-install
cmake --build /new/r1-sdk-build --parallel 2
ctest --test-dir /new/r1-sdk-build --output-on-failure
cmake --install /new/r1-sdk-build
/new/r1-sdk-install/bin/r1_fsmn_pcm --research \
  /new/r1-sdk-install/share/kws-r1/model.f32le \
  /new/r1-sdk-install/share/kws-r1/frontend.f32le /owner/input.pcm
```

Input is raw **mono 16000-Hz signed PCM16 little-endian**, not a WAV file. `-`
reads stdin. The installed C runtime has no Python, Torch or network dependency.
Omit `--model-package` for source-only preparation; it cannot synthesize weights.
Preparation requires a fresh directory and never launches inference. Failed
partial output is retained. The CMake project produces static/shared libraries,
a command-line consumer and a mock-only CTest. Cross-compilation accepts a normal
CMake toolchain file; host build success is not target-toolchain validation.

## Streaming API

`r1_sdk.h` exposes create/feed/gap/finish/destroy and size queries. Initialization
and destruction may allocate; feed, gap and finish do not allocate or perform
file I/O. A callback must not re-enter, destroy or mutate its handle. The CLI's
stdout printing is outside the library's realtime data path; an embedded callback
should copy the event into an application-owned bounded queue instead.

Feed includes an absolute **byte offset** and accepts at most 6400 bytes. Odd
partitions are supported; EOF rejects an incomplete PCM16 sample. Wrong offsets,
oversized blocks and invalid arguments consume zero bytes. A core/callback error
is fail-stop until explicit gap/reset. `r1_sdk_gap(next_sample)` rejects backwards
positions and resets the acoustic cache, CTC repeat state, prefix and cooldown;
it does not insert zero PCM. Every segment has a new epoch. No EOF padding or
synthetic context frames are emitted.

The policy is unchanged from the original CLI: argmax over all 2601 classes,
standard CTC repeat/blank handling, both exact four-token keywords, maximum span
150 model frames, same-keyword cooldown 50 frames. There is no probability
threshold or claimed natural FAR. `required_end_sample` describes the input
dependency; `delivery_end_sample` includes transport batching. Neither is a
human-labelled keyword end or measured keyword-end latency.

## Actual integration check (2026-11-10)

On the fixed 134 original PCM records (380.0800625 seconds; 12534 model frames):

- Fresh GCC and Clang reference64 captures match the authenticated archived x86
  reference **bit for bit**, including all 5,013,600 features and 32,600,934 logits
  per comparison. All 134 archived capture hashes reproduce.
- All 134 complete greedy sequences match the saved original R1 observations.
- 640-byte and alternating 1/37/639/1280-byte SDK feeds produce identical proposal
  IDs, model frames and source-dependency clocks; delivery clocks are checked
  separately because chunking legitimately changes delivery time.
- Four predefined original records pass ARMv7 **archived core** execution under
  QEMU against the host. This does not validate the newly added SDK on ARM.
- 21 invented decoder/transport groups pass GCC, Clang and ASan/UBSan. Four real
  PCM cases pass the sanitized SDK CLI, four pass the installed CLI, and four
  split-stream/gap probes observe zero linked hot-path allocations/frees.

Development remains the same exposed synthetic weak-label cohort: K1 and K2
respectively 10/10 positive-row hits and 0/24 negative-row false proposals.
Training remains 96 eligible rows; four unknown train-role rows are unscored.
No new independent test, human gold, natural continuous FAR or acoustic-end
latency is established. Do not multiply independent support by replay count.

x86 state is 249384 bytes plus 3028860 weight bytes, not total process RSS or
SSC305 RAM. The 134 SDK feeds took 10.0000 seconds for 380.0801 seconds of audio
on this host, excluding per-record create/destroy; this is not target performance.

The numerical rule remains atol=rtol=1e-8 with maximum feature 1e-8 and logit
1e-6 under the **separate reference64 contract**. Historical FP32 compatibility
remains FAIL_UNCHANGED. Independent acoustic evaluation, calibrated detection,
INT8, new-wrapper ARM ABI execution and physical SSC305 measurements remain open.
Raw audio, per-row observations, model weights and private feature matrices are
not part of this source change. Original notices and research-use boundaries
continue to apply; package construction grants no additional rights.
