# R1 strong FSMN CPU optimization — opt-in exact-arithmetic candidates

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

Two opt-in candidates compute **two or four adjacent non-precise output
rows together**, load each input group once, and update eight (`block2`) or
sixteen (`block4`) row-lane accumulators per block in the same output order.
Block2 reduces FP register pressure relative to block4. It never groups across a
precise row; residual rows use the original `par4` loop. This can reduce
repeated input loads, loop overhead and head dispatch, but CPU speedup depends
on architecture and compiler. It is **not** NEON FP32 or a precision change.

The materializer refuses source drift before writing any output and does not
overwrite the existing SDK. It reproduces the source authentication of the
borrowed and qhead8 generators and records both source hashes. For each variant, the only changed
generated SDK C file is `core/r1dfsmn.c`. Other generated changes are
provenance receipts and this README. To generate source-only:

```sh
python3 -I -S -B research/r1_stream_c/cpu_opt/materialize.py \
  --repo . --output /fresh/r1-cpu-block2-sdk --kernel block2
cmake -S /fresh/r1-cpu-block2-sdk -B /fresh/r1-cpu-block2-build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build /fresh/r1-cpu-block2-build --parallel 2
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
   `CLOCK_PROCESS_CPUTIME_ID` timing for each candidate is diagnostic, *not* a product speedup.
2. **ARM source check**: compile/link both `block2` and `block4` Cortex-A32
   ARMhf SDKs, and require real ARM/QEMU execution of invented bitwise head
   fixtures plus correct ELF symbols/ABI, with no model execution. ARM intrinsics are deliberately **not** assumed.
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

## Offline SSC305 CPU evidence comparison (new, opt-in)

The read-only [device_compare.py](device_compare.py) evaluates **already
collected** private target-board JSON receipts from the existing
[\`target_resource_bench.c\`](../quant_head8/target_resource_bench.c).
It never opens the original PCM or model and never launches a model forward.
The existing target resource benchmark uses the **copied** SDK constructor for
all variants to isolate steady-state CPU cost (rather than comparing
constructor memory behavior). Each target library must be built separately
from the identical saved input model, using \`par4\`, \`block2\` or \`block4\`
materialization and the actual SSC305 BSP-compatible compiler/sysroot.

Prepare a **new private** directory with \`PLAN.json\` and 27 or more raw
resource receipts. The plan schema is \`r1-cpu-board-rotated-plan-v1\` with:
- Exact \`model_sha256\`, \`frontend_sha256\`, \`pcm_sha256\`, and
  \`pcm_bytes\`; \`input_mode: offline_pcm16k\`.
- \`binary_sha256\` map containing three distinct operator-computed hashes
  for \`par4\`, \`block2\`, \`block4\`.
- \`board\` recording operator-reported \`id\`, \`firmware_sha256\`,
  \`machine\`, \`governor\`, \`frequency_khz\`, and \`affinity\`.
- \`rounds\`: 9–30 complete counterbalanced rounds in triples,
  with exact orders \`[par4, block2, block4]\`,
  \`[block2, block4, par4]\`, \`[block4, par4, block2]\`;
  each round's \`receipts\` maps the three variant names to
  \`{"path": "receipts/00-par4.json", "sha256": "<actual file SHA256>"}\`
  and corresponding paths for the other variants.

On the **actual** owner-controlled target, fix CPU governor, frequency,
affinity, AFE/PCM input bytes, BSP/rootfs and thermal conditions and rotate
the variants on the same hardware. Save **unmodified** stdout JSON receipts
using exclusive filenames and preserve an independent SHA256 input/ELF index
privately. Avoid live microphone claims when the input is an offline file.

\`\`\`sh
python3 -I -S -B research/r1_stream_c/cpu_opt/device_compare.py \
  --root /private/r1-ssc305-cpu-evidence \
  --output /private/r1-ssc305-cpu-comparison.NEW.json
\`\`\`

The analyzer validates exact counts, measured CPU/RTF arithmetic, balanced
round order, immutable SHA256 bytes, no repeated receipt content, same model
frame/event totals and conservative power-of-two p95 feed histogram bounds.
It screens for at least 5% median **paired** process-CPU reduction, wins in
all but at most two runs, and no p95-upper-bin regression in all but at most
two runs. A passing \`RESOURCE_SCREEN_PASS_NOT_QUALIFIED\` is **not a deployment
decision**. Even though all receipts include \`source-hash-checked-by-sdk\`,
neither hardware identity, toolchain/binary hash claims nor per-frame model
outputs can be **independently authenticated by this receipt-only analyzer**.
Public GitHub must retain only generic code and synthetic tests; no private
model, PCM, serial number, voice, row-level trace or raw receipts.

Before considering a candidate: separately execute exact all-frame F/E parity
against the original owner-held 134-trace manifest, then measure continuous
real AFE wake-end delay, physical-board process CPU/p95/p99/RSS/power/thermal,
and independent natural FAR/hour and real-human accent FRR. Unknown/absent
hardware evidence remains NOT_RUN, not a zero. Existing FP32-vs-Torch numerical
gate remains FAIL_UNCHANGED.

The six new invented-receipt contract tests are executed via the existing
\`cpu_opt/test_contracts.py\` required CI entry. They synthesize **no PCM or
weights**, and verify both pass/fail outcomes, source identity forgery
rejection, symlinks, duplicate receipt payloads/JSON keys, malformed latency,
and CLI no-overwrite behavior. No new workflow, scheduled job, new CI
dependencies or private runner quota are introduced.
