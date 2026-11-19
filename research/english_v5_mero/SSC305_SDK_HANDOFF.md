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

## Model provenance (not in the SDK)

- Frozen real **43,876-param / 256-update FSMN-CTC**: kws-data run [44495594627](https://github.com/jiying2007/kws-data/actions/runs/44495594627), artifact `14514228406`, original ZIP SHA256 `91b203f05041d92e7ffb1b748f94959ce19af6673b91dbc2144920815058d702`.
- Repack on the build host ONLY using `pack_mero43.py` with that original immutable ZIP, then separately authenticate the output pack SHA before provisioning or loading it. The fixed C11 package is **216,960 bytes**. Never regenerate weights from arbitrary `*.pt`, use self-declared unlabeled model weights, or silently accept a new model package. The C loader validates package format/finite tensors but is **not** a cryptographic SHA verifier.
- Verified software numerical chain: [PyTorch↔x86 C11 run 44541193521](https://github.com/jiying2007/kws-data/actions/runs/44541193521) maximum absolute delta `0.0001220703125` on the 10 frozen prospective original PCM, and [x86↔Cortex-A32 QEMU run 44542782505](https://github.com/jiying2007/kws-data/actions/runs/44542782505) with max absolute delta `0.0`, full frame-token and event-frame agreement.

## Device completion gates (not met by this package)

An SSC305 integrator still needs the **actual final board microphone / audio front-end**, continuity/reset handling, sound-echo treatment and per-stream bounded memory accounting. Retain physical-board CPU and RAM measurements, P95 end-to-end inference/callback delay and long-run stability, independent real-human 3–5 m FRR, natural continuous FAR/h and untouched CAL/LOCKED test proof. The above ten synthetic development originals do **not** estimate real FRR or FAR/h, and QEMU is not an SSC305. This target is a valid *integration research SDK*, **not a released/quality-qualified wake-word model**.

No Work/Codex, private repository, paid external ASR/TTS or recurring monitoring job is needed for this standalone build.
