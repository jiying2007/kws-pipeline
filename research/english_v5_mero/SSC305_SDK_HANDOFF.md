# Hey Mero 43K — standalone C11 SDK handoff

This directory is the single-English-keyword research runtime, not the historical two-keyword head. Only **Hey Mero** is active. It is an independently built library, with **no change to the existing shipping `KwsPipeline::core`**.

## Offline compiler / application integration

```sh
cmake -S research/english_v5_mero -B /tmp/mero43-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DHEY_MERO43_BUILD_TESTS=ON \
  -DHEY_MERO43_BUILD_HOST_TOOL=OFF \
  -DCMAKE_INSTALL_PREFIX=/opt/heym43-sdk
cmake --build /tmp/mero43-build --parallel 2
ctest --test-dir /tmp/mero43-build --output-on-failure
cmake --install /tmp/mero43-build
```

Link in a consuming CMake application:

```cmake
find_package(HeyMero43 0.1 CONFIG REQUIRED)
target_link_libraries(my_16khz_pcm_application PRIVATE HeyMero43::stream)
```

The installed public headers are `heym43/mero43_stream.h` and `heym43/mero_ctc.h`. No Python runtime or model training is required on SSC305. Cross-compile with the project-specific Cortex-A32 ARM hard-float sysroot/toolchain; do not claim actual-board execution from a cross-compile. The core requires a caller-owned model arena and stream state; query `en_v5_m43_model_bytes()` and `en_v5_m43_stream_bytes()` rather than hardcoding structure sizes. Inference uses no heap allocations, file I/O or threads in the core. The caller supplies **16 kHz mono little-endian signed PCM16** with monotonic sample indices, consumes events by callback, and calls reset or discontinuity explicitly on audio gaps.

## Live post-AFE input on POSIX systems

Set `-DHEY_MERO43_BUILD_LIVE_TOOL=ON` in the SDK build to install `bin/heym43-live`. Feed one continuous session of 16 kHz mono signed little-endian PCM16 from the board's post-AFE output. The adapter reads raw bytes from stdin and writes one JSON line per complete wake event to stdout. The capture application owns the microphone, channel selection, sample rate and AFE.

```sh
: "${B0_PACK:?path to the authenticated B0 pack}"
: "${MAX_GAP_SAMPLES:?maximum inter-token gap in samples}"
: "${COOLDOWN_SAMPLES:?refractory period in samples}"
: "${POST_AFE_FIFO:?raw PCM FIFO from the capture application}"
heym43-live "$B0_PACK" "$MAX_GAP_SAMPLES" "$COOLDOWN_SAMPLES" < "$POST_AFE_FIFO"
```

Verify the B0 pack against SHA-256 `ff5d0868c035bf5b1c696115eb62f42c36ddbd3a62781207b1340f5e7d1f9d9b` before starting the adapter. The two decoder settings are mandatory. Product settings require separate calibration. A short read can split a sample. The adapter retains its first byte until the second arrives, rejects a trailing half-sample at EOF and reports sample, frame and event totals to stderr at a clean EOF.

The output fields are `event`, `keyword_id`, `frame_index`, `available_after_sample` and `callback_monotonic_ns`. Frame indices begin at zero. The sample clock counts consumed input samples, and the monotonic timestamp comes from the callback's own clock. Physical latency also requires a time-aligned, independently observed word ending. Keep stdout drained because a blocked consumer stalls audio ingestion. A raw pipe cannot signal a dropped interval. End the process on capture loss and start a new process/session when capture resumes, or use the C API's explicit discontinuity operation in the capture application.

## Model provenance (not in the SDK)

- Frozen real **43,876-param / 256-update FSMN-CTC**: kws-data run [44495594627](https://github.com/jiying2007/kws-data/actions/runs/44495594627), artifact `14514228406`, original ZIP SHA256 `91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702`.
- Repack on the build host ONLY using `pack_mero43.py` with that original immutable ZIP, then separately authenticate the output pack SHA before provisioning or loading it. The fixed C11 package is **216,960 bytes**. Never regenerate weights from arbitrary `*.pt`, use self-declared unlabeled model weights, or silently accept a new model package. The C loader validates package format/finite tensors but is **not** a cryptographic SHA verifier.
- Verified software numerical chain: [PyTorch↔x86 C11 run 44541193521](https://github.com/jiying2007/kws-data/actions/runs/44541193521) maximum absolute delta `0.0001220703125` on the 10 frozen prospective original PCM, and [x86↔Cortex-A32 QEMU run 44542782505](https://github.com/jiying2007/kws-data/actions/runs/44542782505) with max absolute delta `0.0`, full frame-token and event-frame agreement.

## Device completion gates (not met by this package)

An SSC305 integrator still needs the **actual final board microphone / audio front-end**, continuity/reset handling, sound-echo treatment and per-stream bounded memory accounting. Retain physical-board CPU and RAM measurements, P95 end-to-end inference/callback delay and long-run stability, independent real-human 3–5 m FRR, natural continuous FAR/h and untouched CAL/LOCKED test proof. The above ten synthetic development originals do **not** estimate real FRR or FAR/h, and QEMU is not an SSC305. This target is a valid *integration research SDK*, **not a released/quality-qualified wake-word model**.

No Work/Codex, private repository, paid external ASR/TTS or recurring monitoring job is needed for this standalone build.

## Latest candidate quality closure — B20 (2026-11-24)

The B17 paired model checkpoints were evaluated once on the new B20 synthetic panel together with the fixed B17 natural speech panel in [kws-data run 45337450827](https://github.com/jiying2007/kws-data/actions/runs/45337450827). The run completed all **1,008 model forwards**: 84 admitted originals × three fixed gains × four frozen model roles. The original logits and attempt journals were independently read back without new inference.

The numerical result is **NO_PROMOTION**. Relative to the paired control, the challenger lost one positive condition and added three natural-speech false-trigger conditions. It recovered none of the four B9-recoverable original/gain cases across three source groups. Natural ordinary-speech events at 0.75 gain were 7 versus B0's 6. The original opening reader rejects this result; B20 replication and candidate engineering validation remain closed.

The original producer ZIP omitted 32 `.github` files. A separately identified supplement binds each missing file to the real producer Git tree, Git blob and SHA256; the original artifact is unchanged and is not described as complete. All 17 original input archives and all 120 original source rows were authenticated, and the 84 admitted rows exactly reproduce the original admission. See [the B20 evidence directory](https://github.com/jiying2007/kws-data/tree/main/research/english_v6_b20_results_20261124) and `V6_QUALITY_EVIDENCE.json`.

The installed integration baseline remains **B0**. No B17/B20 candidate is released or substituted into this SDK. This result is development evidence; it provides no measured continuous FAR/h, field 3–5 m FRR, or physical SSC305 result.
