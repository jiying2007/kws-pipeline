# Optional sherpa PCM research adapter

[简体中文](README.zh-CN.md)

This independent research entry point reuses sherpa-onnx1.13.8's official C ABI with a small PCM lifecycle layer. It does not change the product library, default model, root CMake project, shipping qualification or existing runtime contracts. It is an effect-reference/backup deployment candidate; compact native models remain the CPU-first research priority.

`pcm_kws.h` exposes create/feed/callback/finish/reset/destroy. Input is mono16k signed PCM16; a320sample staging buffer makes detection availability independent of caller partition. CPU1, feature80, maxpaths4, score1, threshold0.25, trailing blank1 are fixed to the observed recipe. Create loads once; reset creates a fresh stream without reloading weights; finish flushes partial input and signals EOF without adding silence. Callback strings expire when the callback returns. IDs1/2 map the two target phrases;0 means an unexpected keyword. Calls are single-threaded and non-reentrant; callbacks must not throw, reenter, or destroy the instance.

The adapter has no explicit audio-feed filesystem I/O. ORT can allocate or fault pages; this layer does not satisfy the product runtime's allocation-free realtime guarantee. Exceptions are caught at exported boundaries; argument/state errors return codes and optional error text. Upstream aborts/OOM termination, invalid pointers and hardware failures are not recoverable. Model/token/keyword digests are the caller/package's validation responsibility. No beamforming, resampling, ALSA, PortAudio or WebSocket is introduced.

## Model-free tests, no downloads

```
python3 research/sherpa_pcm/build.py --stub --output /tmp/kws-sherpa-stub
python3 research/sherpa_pcm/verify_dependencies.py
python3 research/sherpa_pcm/tests/test_parity_guards.py
```

The stub implements only lifecycle test behavior against the vendored exact official header. It is not speech inference and cannot produce quality/performance evidence. Tests compile a C caller, validate fixed settings, exercise partition staging, partial-input reset, EOF and invalid states, and repeat creation/destruction. The dedicated workflow runs these offline tests. The test stub never enters the default build.

## Real runtime build, explicitly supplied dependency

`dependencies.lock.json` binds exact header source/SHA, official model snapshot/hash URLs, decoder settings, and previously inspected x86 runtime hashes. The vendored header is Apache2 with its retained license. Model data rights require their own review. No model, shared library, wheel or executable is stored in this source patch. Do not use the fake runtime for acoustic tests.

```
python3 research/sherpa_pcm/build.py --output /tmp/kws-sherpa-real \
  --library /absolute/reviewed/lib/libsherpa-onnx-c-api.so
```

Supply the matching ONNX Runtime library and its dependencies through an explicit package/runtime search path. Current inspected host pairing is sherpa1.13.8 with ORT1.28.2. ABI compatibility and packaging remain an integration responsibility; there is no automatic download or fallback.

Optional observed-clip parity, using already-authorized local assets/data/reference:

```
python3 research/sherpa_pcm/tests/real_clip_parity.py \
  --library /tmp/kws-sherpa-real/libkws_sherpa_pcm.so \
  --build-receipt /tmp/kws-sherpa-real/build-receipt.json \
  --model-dir /pinned/models --keywords /pinned/keywords.txt \
  --data-root /pinned/kws-data --reference /local/readback.json \
  --output /local/parity-result.json
```

This test-only Python driver verifies WAV+PCM hashes, then42 original clips under fixed320, irregular and whole-clip caller partitions. It requires the established readback reference and keeps source labels out of inference. The corresponding local candidate test passed126 executions with exact keyword/availability-samples/EOF parity, plus lifecycle checks. Output remains local; publication requires the intended data-sharing authority. Python is not a native library dependency.

## Evidence and guarantee boundaries

The earlier separate research CLI measured19/20 positive hits and zero22confusable events, CPU RTF0.018488 and whole-process RSS60,204KiB on an x86 EPYC host. RSS includes5,027,840B preloaded audio, runtime/weights and instrumentation. That CLI's installed components totaled37,938,029B: model/token5,737,545B, C API5,120,672B, ORT27,026,609B, runner53,128B and keywords75B, excluding system libraries/package overhead. These are measured x86 components, not ARM package estimates. That CLI is not this extracted adapter; the adapter itself is not performance-profiled. Observed synthetic clips establish neither fresh-holdout quality nor continuous FAR nor far-field performance.

SSC305 is user-identified dual-core Cortex-A32. Upstream ARM32 sources provide an armv7-a hard-float/NEON route, but actual BSP libc/ABI/sysroot/compiler/operator compatibility remains to be verified. Preparing host software and synthetic/open-data validation does not depend on board access. Acceptance on the board later checks CPU deadlines under concurrent services, I/O/page faults, DMA/capture integrity and long-run memory. No x86 speed extrapolation is made.

The standalone build uses installed GCC/G++ and Python standard-library orchestration; it does not require CMake. The root repository CMake/ctest suite was not run in this environment because those tools are unavailable. The research entry does not join the root product build. The real-library recipe currently verifies the exact inspected x86 library digests; target builds require a separately reviewed dependency lock rather than bypassing verification.

Before native loading, the optional parity driver requires the exact approved reference SHA, a matching non-stub build receipt/source lock/library, and unchanged pinned runtime dependencies. It rejects absolute/traversal/symlink recording paths and existing output files. Ten model-free negative/positive guard tests cover these checks. Build receipts are provenance, not cryptographic attestations against maliciously forged local artifacts.
