# R1 strong FSMN CPU optimization — opt-in exact-arithmetic candidate

This is a **research-only, source-bound experiment** extending the already merged
`qhead8/par4` borrowed-view SDK. The FP32 source, pretrained checkpoint,
quantized 1,960,140-byte model, frontend, decoder, default SDK and original
copied constructor are not edited. There is no ASR/TTS, training, new audio,
periodic patrol or paid inference.

## CPU change

The original head computes 2,601 outputs, each with a 140-term weighted
reduction. Five precise blank/keyword rows use the untouched serial FP32
weight/double-accumulation path. The remaining rows retain the existing four
separate double accumulators and the reduction order `(s0+s1)+(s2+s3)`.

The candidate computes four adjacent **non-precise output rows together**,
loads each four-element input group once, and updates the 16 row-lane
accumulators in the same per-output index order. It never groups across a
precise row; residual rows use the original `par4` loop. This can reduce
repeated input loads, loop overhead and head dispatch, but CPU speedup depends
on architecture and compiler. It is **not** NEON FP32 or a precision change.

The materializer refuses source drift before writing any output and does not
overwrite the existing SDK. It reproduces the source authentication of the
borrowed and qhead8 generators and records both source hashes. The only changed
generated SDK C file is `core/r1dfsmn.c`. Other generated changes are
provenance receipts and this README. To generate source-only:

```sh
python3 -I -S -B research/r1_stream_c/cpu_opt/materialize.py \
  --repo . --output /fresh/r1-cpu-block4-sdk
cmake -S /fresh/r1-cpu-block4-sdk -B /fresh/r1-cpu-block4-build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build /fresh/r1-cpu-block4-build --parallel 2
```

Optional owner-held `--model ORIGINAL_MODEL --frontend FRONTEND` packs only
the same authenticated, pre-existing R1 model. Never push generated
weights, PCM, trace rows or private manifests into this public repository.

## Bounded proof and measurement stages

1. **CI source check**: build source-only SDK with GCC/Clang, compare the
   generated SDK against unmodified borrowed source, and exercise the extracted
   original and candidate head using invented, deterministic finite arrays.
   Require bitwise equal 2,601 output values across zero, alternating,
   near-zero, high-dynamic-range and random inputs with `-O2` and `-O3`,
   `-fno-fast-math` and `-ffp-contract=off`. A synthetic head-only
   `CLOCK_PROCESS_CPUTIME_ID` timing is diagnostic, *not* a product speedup.
2. **ARM source check**: compile/link a Cortex-A32, ARMhf library using the
   existing repository ARM toolchain check; require correct ELF symbols/ABI,
   no model execution. ARM intrinsics are deliberately **not** assumed.
3. **Owner-only inference**: using the same saved 134 PCM inputs, source-verified
   original model and existing saved traces, compare every original F/E trace
   byte-for-byte against the new host and ARM/QEMU candidate **before promotion**.
   Reuse the previous candidate validation evidence and only create new
   bounded candidate receipts when the owner input artifacts are available.
   There is no fallback if assets/toolchain are absent.
4. **CPU qualification**: measure `process CPU time / audio second`,
   frame p95/p99, live audio front-end + head + decoder CPU time, and full
   stream RSS under a pinned governor/CPU frequency; compare baseline and
   candidate on the **physical SSC305** with target BSP/sysroot, and capture
   power/temperature. Merely compiling on an ARM cross toolchain or running
   ARM/QEMU cannot establish CPU savings on a real device.

**Promotion is blocked** until real-model exact frame/event parity, credible
CPU reduction and the separate V3 independent acoustic/device gates are
verified. The historic FP32-Torch numerical gate remains `FAIL_UNCHANGED`.
Do not claim an optimization from source restructuring or invented fixture
measurements alone.
