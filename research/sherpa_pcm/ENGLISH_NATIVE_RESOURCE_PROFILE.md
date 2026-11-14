# Isolated native C-only CPU/RSS and 20 ms feed-window profile

## Why

Prior measured pretrained English INT8/c16 host C inference is real and
reproducible, but its Python driver buffered source WAV/FLAC arrays. In the
61-minute LibriSpeech experiment the 434,196-KiB **process** peak was caused
in large part by the runner's Python/audio structures, so claiming this as
the native KWS model footprint would be misleading.

This new research profile launches a *standalone C executable as a separate
Linux process*, links the actual existing optional PCM KWS adapter and uses
the exact official pretrained model. The child process does NOT import
Python, NumPy, TTS, ASR, Torch, audio-generation code or read huge buffers.

## Immutable inputs and measured scope

- Original 63 already-generated Qwen English mono16 kHz PCM16 WAVs from
  original ZIP SHA256
  f180004a27ea14c0575f4dd0e2a8476ca048442c76c2dcc186ede6d731cdffcc,
  with source review JSON SHA256
  89f20b7ac5a672840013b3cf780422dcf68451b6f5d618e8f494a0e5f5dc6b12.
  Parent validates 63 WAV + PCM SHA256 pairs, roles and exact 2,101,760
  PCM samples BEFORE creating any child WAV source view. No synthesized
  sample, source editing, train label admission or additional provider call.
- Original Sherpa-ONNX Zh-En 3M INT8/chunk16 official model archive SHA256
  68447f4fbc67e70eee3a93961f36e81e98f47aef73ce7e7ca00885c6cd3616a6.
  Same published source official HEY_MERO keyword, score1, threshold.25,
  CPU1, feature80, path4 and one trailing blank; no model adaptation.
- Original official Sherpa 1.13.8 x86_64 runtime archive SHA256
  c0bdb7907d3a74bba1d55d22bf4d9fa75586cf1530614ebe88a27b9118e015c4.
  Exact original C ABI/ORT inner libraries and the compiled existing C
  adapter source are checked by prior readback before native profiling.
- Fresh child process loads the real C model once; it resets stream state
  between 63 original input WAVs, without adding any silent PCM.
  Original previously observed native no-tail event count remains **5**;
  this is not a new model-selection or FAR/FRR experiment.

## Measurements, not invented values

The independent C executable records actual Linux /proc/self/status
VmRSS before model initialization, after model loading and after original
PCM replay, and RUSAGE_SELF maximum RSS. These are **whole C process**
measurements: dynamic linker, ORT, Sherpa, adapter, model and other runtime
allocations contribute. The difference across create is an indicative
constructor increment, NOT a proof of model tensor storage or peak resident
model memory. The child does not hold the parent's Python/source arrays.

Each call to pcm_kws_feed consumes no more than 320 PCM samples (20 ms
of audio). The child measures CLOCK_MONOTONIC and PROCESS_CPUTIME_ID
elapsed times around **each real C feed call**. Sorted observed values
give p50/p95/p99/max and total feed CPU/wall time; deadline misses mean
a host feed call exceeded its input PCM duration. The model's create time,
WAV disk reading, callback scheduling, host CPU contention, device audio
I/O, DSP prefilter, audio task scheduling and E2E end-of-word latency are
outside these feed-call windows. This is an **offline accelerated replay**,
not an actual hard-real-time audio ingestion schedule.

C-native process RSS is not equivalent to a SigmaStar SSC305 BSP RSS;
a Cortex-A32 CPU requires its own model binary, toolchain, sysroot, C++
ABI, OS scheduling and power/profile testing. Do not convert host RTF to
target microseconds by an arbitrary CPU frequency ratio.

## Reproduction

The already available public native host C build script uses exact pinned
release hashes, and the parent must use the original WAV artifacts:

    python3 research/sherpa_pcm/build_official_x64.py \
      --runtime-archive /restricted/official-host.tar.bz2 \
      --output /restricted/actual-native

    python3 research/sherpa_pcm/tests/english_native_profile_once.py \
      --native-library /restricted/actual-native/libkws_sherpa_pcm.so \
      --build-receipt /restricted/actual-native/build-receipt.json \
      --original-audio-zip /restricted/original-qwen63.zip \
      --original-review /restricted/english63-three-evidence-review.json \
      --pretrained-model-archive /restricted/official-zh-en3m.tar.bz2 \
      --output /restricted/native-only-profile.json

All intermediate original WAVs are in the runner's private temporary
directory and are deleted after execution. C source compiles model-free
under the existing Sherpa source CI; the actual separate-process test is
a one-shot public GitHub Actions job using original source artifacts. No
model weights, runtime .so, original audio or credentials are committed.

## Research-only stop conditions

This is CPU/RSS measurement for the x86_64 host ONLY. It does not certify
real SSC305 single-core capacity, target p99 deadline, target power/heat,
allocation-free callback properties, independent human wake FRR,
continuous household false-wake FAR/hour, acoustic accent acceptance,
far-field 3–5m or verified lexical end-of-word trigger delay. The original
Chinese wake runtime remains unchanged. No model promotion, threshold
selection or synthetic-speech data regeneration is allowed from this step.


## Actual original-source C-only profile (2026-11-14)

Source-owning [Actions run 43569419467](https://github.com/jiying2007/kws-data/actions/runs/43569419467)
**completed successfully**. Its original machine artifact ID 14103930651,
ZIP SHA256 f349fd8c8f55533a8b1cb8e9545bf481c69fb7ac7b1df4a92d28032bf644c9e8,
was independently downloaded, and original JSON (2,587 bytes) SHA256
676fe3304d916d5612d91a31790320ec77918ede2a3b42ffeb6c994be689b012
was read back. Source and code SHA256s, model/runtime source and exact
measurements are retained in ENGLISH_NATIVE_RESOURCE_RESULT.json.

The independent Linux x86_64 C process loaded the unchanged official
Zh-En INT8 chunk16 model, replayed all **63 original Qwen synthetic WAVs**
(2,101,760 source PCM samples; 131.36 seconds) in 320-sample C calls,
without adding silence. It reproduced **5** actual HEY_MERO ID3 events,
zero unexpected/EOF-only events. No Python interpreter/audio buffers were
present in the *profiled child process*.

| Original measurement, one host process | Measured value |
| --- | ---: |
| Whole C process RSS before model creation | 17,568 KiB |
| Whole C process RSS after model creation | 52,340 KiB |
| Whole C process RSS after all clips / peak | 57,720 KiB |
| Approx. RSS increase at create (not tensor memory) | 34,772 KiB |
| Original model encoder+decoder+joiner disk bytes | 5,446,114 B |
| Native C feed callbacks, <=320 PCM samples each | 6,568 |
| Actual wall feed latency p50 / p95 / p99 / max | 0.014954 / 3.423519 / 3.561917 / 5.971256 ms |
| Actual feed CPU latency p50 / p95 / p99 / max | 0.014696 / 3.422283 / 3.560257 / 5.970135 ms |
| Total C feed CPU time | 1.339160119 sec |
| CPU-seconds / original audio-second (feed calls only) | 0.010194581 |
| Host C feed calls exceeding their input-audio duration | 0 |

Important: the above feed calls were executed in a fast, **offline** loop.
Their 20ms per-chunk comparison is not a real audio device scheduling
guarantee. RSS values measure the **whole host child process**, not
the 5.45MB model file sizes or exact isolated tensor buffers; true
memory includes Sherpa, ONNX Runtime, dynamic libraries, allocator state,
model initialization and overhead. No native target/SSC305 performance was
measured. No second benchmark run occurred; there are no multi-run
confidence bounds on host p99 or the process RSS.

The prerequisite data-owned one-shot PR #119 is closed without merging
after source-bound measurement readback, so its execution-only workflow
does not become unnecessary permanent CI. Formal model qualification is
still blocked by target SSC305 BSP/sysroot runtime compatibility,
target ARM CPU/RSS/p99 under actual device workload, independent
microphone wake-word positives/accent review, real continuous household
background FAR and verified wake-word-end-to-event latency.
