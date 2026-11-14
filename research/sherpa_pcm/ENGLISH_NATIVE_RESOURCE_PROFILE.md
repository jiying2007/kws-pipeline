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
