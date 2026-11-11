# Strong R1 FSMN target resource profiling (research-only)

This tool profiles the **same** pretrained R1 SDK variants: FP32 reference64, compact head-INT8 scalar and compact head-INT8 par4. It intentionally does not depend on the retired `.kwm/.kwk` board benchmark format. The instrumented C source is [`target_resource_bench.c`](target_resource_bench.c).

## What it actually measures

A **Linux user process**, supplied with legitimate owner-held frozen R1 weights, frontend and **16 kHz mono signed PCM16LE** input, reports measured:
- actual model initialization wall time (model loading and hash checks);
- per-640-byte offline file-feed wall times (max and bounded logarithmic p95 **upper bound**, not exact p95);
- total file-feed wall time, process CPU time, audio seconds, RTF and CPU-seconds/audio-second;
- Linux `getrusage(RUSAGE_SELF).ru_maxrss` peak RSS in KiB, frame count and complete-word event counts;
- kernel release and userspace machine name.

**Important limitations:** This is offline file feeding, **not** real microphone/AFE capture or continuous always-on latency. Instrumentation overhead and file I/O contribute to times; it is not the CPU cost of an isolated acoustic kernel. `ru_maxrss` is a per-process peak, not the whole robot's memory. The power, stack high-water, calibration, true word-end p95 and physically certified SSC305 identity are not measured and remain explicitly null/false. Cross-architecture QEMU measurements are not SSC305 throughput/power evidence.

## Owner-controlled board experiment

In the **owner-controlled** build environment, obtain an actual SSC305 firmware-compatible SDK sysroot and compiler, and the three matching R1 C SDK static libraries/headers. Do not assume generic QEMU ARMhf ABI equals the SSC305 firmware ABI. For each of the three variants, using its matching header/lib and corresponding weights:

```sh
# Set these paths to the actual variant; do not infer them from example names.
"$TARGET_CC" -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -I"$SDK_INCLUDE" \
  research/r1_stream_c/quant_head8/target_resource_bench.c \
  "$SDK_STATIC" -lm -o "$OUTPUT_BENCH"

# On the actual target: copy only the authorized binary, model/frontend and audio.
# Run each variant with exactly the same POST-AFE / PCM identity and CPU governor.
"$OUTPUT_BENCH" "$MODEL" "$FRONTEND" "$POST_AFE_PCM" > "$PRIVATE_RECEIPT"
```

### Optional immutable-model mapping, same resource probe

With the **generated borrowed-view SDK header and corresponding static library**
(`FSMN_MODEL_VIEW_ALIGNMENT` exposed), the same C resource benchmark has a
Linux opt-in `--mmap` mode. The default, three-argument invocation keeps its
original copied-model behavior. The frozen reference64 header lacks the view
constructor and explicitly **rejects** `--mmap` rather than pretending that
copying was an optimized run.

```sh
# Build with each generated borrowed qhead8/par4 or cpu_opt block2/block4 SDK:
"$TARGET_CC" -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  -fno-fast-math -ffp-contract=off -I"$BORROWED_SDK_INCLUDE" \
  research/r1_stream_c/quant_head8/target_resource_bench.c \
  "$BORROWED_SDK_STATIC" -lm -o "$OUTPUT_BENCH"

# The model must be a verified, immutable, exact-size regular file.
"$OUTPUT_BENCH" "$MODEL" "$FRONTEND" "$POST_AFE_PCM" --mmap > "$PRIVATE_RECEIPT"
```

The mapping path uses `O_NOFOLLOW`/`O_CLOEXEC`, exact regular-file size,
`MAP_PRIVATE|PROT_READ`, the unchanged `fsmn_sdk_create_view` model
SHA-256/layout validation, and SDK destruction **before** unmapping. It does
not eliminate file-backed residency, reduce the actual FSMN math, prove sound
quality, or prevent another process from changing/truncating a writable
underlying file. Prefer immutable rootfs/squashfs model assets. The receipt
now identifies `model_storage` plus caller heap bytes and optional PSS and
private-dirty memory (null if unsupported), without altering the existing
offline feed CPU/RTF/schema fields.

**Do not compare mixed loader modes** in a paired CPU A/B. For `par4`,
`block2` and `block4` choose *all copy* or *all mmap*; the optional-model
storage verifier rejects mixed or forged labels. True physical-board CPU,
power, file-cache pressure, AFE and natural/human accuracy evidence are
separate from CI's invented-only source tests.

Requirements for a comparative run:
1. Preserve the exact `main` source SHA, **binary ELF/build-id and SHA256**, shared/static library identity, rootfs and board revision, actual `uname` / `/proc/cpuinfo`, toolchain/sysroot identities, CPU affinity/governor/frequency and the post-AFE PCM SHA256 and byte count.
2. Run a bounded warm-up and measured series **on the board**, rotating FP32/scalar/par4 order; measure model initialization separately, RTF, p95 upper bound, max, RSS, and process CPU. Note thermal state and other system load; if the board doesn't support a metric leave it **NOT_RUN**, never zero.
3. Save the **raw JSON receipts** and original monitor/power traces privately, with an immutable SHA256 index; then independently recompute it. No raw human PCM, private weight or individual decision logs go to public GitHub.
4. For true target **power** use a real instrument and calibration evidence; for stack high-water use separate stack instrumentation; for continuous FAR/hour use measured uninterrupted negative-audio exposure; for FRR use independently held-out human speech. These are **separate** studies and cannot be inferred from this benchmark.
5. Abnormal PCM sizes, invalid models, early I/O failure, invalid clock/resource observations or SDK callback errors **fail nonzero** without a success JSON. No padding or implicit resampling is performed.

The existing [`docs/TARGET_EVIDENCE.md`](../../../docs/TARGET_EVIDENCE.md) documents independent board/soak evidence governance for the historical engine. Its old `kws_board_bench` accepts `.kwm/.kwk`, not the strong R1 SDK format. Do not relabel its results as R1 measurements. This stand-alone tool does not confer release, deployment, or acoustic admission authority.

## CI scope

The repository CI compiles this **source-only** target runner against the frozen public R1 SDK header using GCC and Clang; no model, original audio, physical board, QEMU inference or private evidence is available in CI. This checks compile and contract boundaries, **not** performance or deployment.
