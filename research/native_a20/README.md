# Isolated A20 native research runtime

This directory preserves a six-class finite-memory keyword-spotting research
candidate. It is independent of the shipping RNN, default build, registry and
product ABI. It does not qualify a model for release or an SSC305/Cortex-A32 board.

The candidate has 390,520 trainable parameters and 800 fixed CMVN values. Model
weights, cache and stage interfaces remain binary32; affine and finite-memory
accumulation use binary64. The frontend keeps the original radix-2 FFT schedule
with two additional 512-double work arrays and certified binary64 twiddles,
then rounds complex outputs to binary32 before power/mel/log processing. There
is no retraining, quantization or shipping-default change in this source port.

## Build and tests

Only Python standard library, a C11 compiler and libm are needed. No installer,
network access, model download or pretrained-model forward is part of these tests.
Choose new output directories outside this source tree:

```sh
python3 -B research/native_a20/build.py --output /tmp/native-a20-build
python3 -B research/native_a20/build.py --output /tmp/native-a20-sanitized --sanitize
```

The tests use invented coefficients/signals. They cover FFT cardinal cases,
rounding ties and subnormals, invalid environment rejection, unchanged
preprocessing, actual short tails, canonical grouping, finite-memory tap/cache
order, model identity rejection, SHA256 vectors, CTC decoder state and exact
whole/ragged stream geometry. ASan/UBSan are exercised separately; leak detection
is disabled and no leak-check pass is claimed. The host workflow runs this offline synthetic suite; a separate job
installs the generic Ubuntu ARM cross compiler and checks compilation/linking only. Production C source is preserved byte-for-byte.

The build emits a research library, a raw PCM CLI, resource-size record and test
receipt. It compiles but never runs the CLI on an archived model or audio asset.
The CLI requires the exact externally materialized payload SHA256
`a80b233d3c958d25f49085f487ef5cde839bd69f803ed057b59bbd20adc6e43d`.
It accepts mono 16-kHz signed PCM16 little-endian raw audio. Its deliberately
minimal JSON output is a research debugging interface, not the shipping API.

### Explicit cross compilation (no target execution)

`--mode native-test` remains the default. Before any generated executable runs or
shared library loads, the driver validates every compiler product against the
running Python process's ELF class, machine and byte order. Native OSABI/version
and ARM EABI/float flags are checked conservatively too. A cross compiler passed
through `--cc` or `CC` fails closed on a mismatched ELF, even if its reported
triple looks native or an emulator is registered on the host. These checks require
an ELF host for native-test mode; they do not prove CPU instruction, dynamic
loader or libc compatibility. Exact native OSABI equality may conservatively
reject otherwise compatible Linux SYSV/GNU-marked outputs; such combinations
need separate evidence, not an automatic bypass.

Use compile-only mode for a target toolchain. All three expected ELF fields are
required, and are checked against the actual outputs, independently of compiler
naming or its descriptive `-dumpmachine` output:

```sh
python3 -B research/native_a20/build.py \
  --mode compile-only --cc arm-linux-gnueabihf-gcc \
  --target-elf-class 32 --target-machine arm --target-endian little \
  --target-flag=-mcpu=cortex-a32 --target-flag=-mfpu=neon-vfpv4 \
  --target-flag=-mfloat-abi=hard --output /tmp/native-a20-arm-compile
```

This builds the same nine objects, shared library, CLI, resource query and five
invented-input test programs. It never executes any of them or uses ctypes to
load the target library, even if the target equals the host. Host-only source
manifest and twiddle-certificate checks still run. Linked ELF files must also
have bounded program headers and valid file-backed executable load segments.
No emulator or target runner is invoked. An existing compiler/sysroot is needed;
the build driver does not install or download anything.

Optional `--sysroot /existing/toolchain/sysroot` applies to every compile/link
command. Repeated `--target-flag=...` accepts only architecture/ABI selection
options (`-mcpu`, `-march`, `-mtune`, `-mfpu`, `-mfloat-abi`, `-mabi`, ARM/thumb and
endianness selectors). Arbitrary compiler/linker flags, response files and
floating-point overrides are rejected. The strict numerical flags stay fixed.
The provided compiler and sysroot must be trusted; this is not a tool sandbox.

The v2 receipt distinguishes compile/link and ELF checks from target execution.
For compile-only, resource-size queries, invented tests and ctypes ABI checks
are `NOT_RUN`; target execution and dlopen counts are zero. There is no
`resource-sizes.json`: host sizes must not be substituted for target measurements.
`--sanitize` can instrument target binaries, but a compile-only receipt never
claims the sanitizers ran. Commands, compiler version/triple, expected ELF fields,
per-file ELF identity/flags, byte sizes and SHA256s are retained. No success
receipt is written on a failed build, and existing output directories are refused.

Run the offline driver guard tests without a cross compiler:

```sh
python3 -B research/native_a20/tests/test_build_driver.py
```

They use invented ELF fixtures and mocked execution/loading to cover cross-CC
rejection, malformed or mismatched ELF, unsafe flags, late artifact failures,
compiler failure/timeout, stale output, unchanged native-test behavior and zero
compile-only execution. The dedicated generic ARM CI job builds only this
research directory, saves static readelf evidence and does not run ARM code.
Generic Ubuntu ARM compilation is not SSC305 SDK/vendor ABI qualification,
target numerical validation, physical Cortex-A32 performance or product release
approval. Those remain separate pending evidence gates.

`export_a20.py --checkpoint INPUT --output NEW_DIRECTORY` exports only the
pinned checkpoint identity, using an already installed PyTorch runtime with
`weights_only=True`. It does not fetch dependencies, train or run the model.
Checkpoint SHA256 is
`5b347c0ce7df6e8ad3649c9186a11acab4b3e35971950934490c511d2b04bc39`;
state SHA256 is
`c05623683b4616badda5535eefb0bfaecde63efbf3bfbceec6e7440a5fc19eb2`.
Model/data publication is tracked separately from source integration; this source
commit does not claim the asset archive has already been published.

## Evidence status and boundaries

The original strict official-FP32 numerical acceptance failed. That failure
remains a failure. The subsequently adopted PCM-derived mathematical contract
is specified in [NUMERICAL_CONTRACT.md](NUMERICAL_CONTRACT.md); it is a different
authority and cannot retroactively pass the earlier gate.

- Replacement eight-signal numerical validation: 1,334 hard checks passed;
  16 whole/ragged native streams were exact
- Exposed twelve-clip regression: 2,208 hard checks passed; 56 chunk records and
  five observed events matched required discrete fields; maximum score error
  against original saved behavior was 6.704e-6
- Official-FP32 raw-logit compatibility diagnostic still failed for four clips
  at twelve coordinates
- A previous eight-signal execution's original raw evidence is permanently
  unavailable. Retained text/hash receipts are not recovered raw evidence.
  The replacement execution has its own complete retained chain

Those are reported historical aggregate results, not outcomes of this PR's
invented-only CI. They provide no FAR/FRR, real-human, final-AFE or board claim.
The public data inventory must identify retained versus missing raw records.

A fixed single-thread x86 host profile executed 48 streams, 224 canonical chunks
and 2,004 model steps in total across four passes (one warmup and three measured),
with matching output digests. Each pass contained twelve clips and 15.4 seconds
of audio. On one AMD EPYC 9V74 CPU, median measured corpus-pass CPU time was
0.165382688 seconds (CPU RTF 0.010739136). A descriptive 168-chunk wall sample
had p50 approximately 3.211 ms and p99 approximately 3.744 ms. Chunk CPU clock
resolution was insufficient: 43 readings were zero and 122 approximately 4 ms;
frequency was unavailable. These are descriptive host observations only.

An invented-coefficient arithmetic proxy completed 5,120 rows with four measured
blocks per implementation: median binary32 0.323799014 ms/row and binary64
0.343566465 ms/row, observed ratio 1.061048522. It compares binary32 four-way
accumulation against binary64 sequential accumulation, so it is not an isolated
precision-width cost or a full-FP32 model baseline. The original launch failed and restored executable mode 0644 was observed;
missing executable bits are a strong causal inference, but the exact failing
errno/entry point was not logged and zero-entry is not proven. That first FAIL
remains recorded. One mode-only recovery to 0744 passed independent audit, for
two cumulative attempts. No numerical source/kernel bytes changed for recovery.

On the measured x86 ABI, caller state was 333,704 bytes plus 1,565,280 bytes of
weights, totaling 1,898,984 bytes; model cache was 22,528 bytes and arithmetic
count was 388,984 MACs/model row. Compiler stack/ABI/FPU and physical A32
performance remain unqualified. Do not extrapolate host timing to a board.

## Layout and provenance

- `src/`: selected FFT64 frontend, PCM adapter, stream API and CLI
- `baseline/precision64/model/`: binary64-accumulation model and verified loader
- `baseline/decoder/`: bounded C11 CTC decoder
- `baseline/native/`: retained original source needed for preprocessing and
  size comparisons, including the strict-FP32 historical model; it is not the
  selected runtime
- `generate_twiddles.py`: input-independent outward interval certificate;
  the full reproducible certificate is omitted in favor of its compact summary
- `SOURCE_MANIFEST.json`: exact public file set and SHA256 identities
- `PROVENANCE.json`: unchanged native-source identities and transport changes

See [NOTICE.md](NOTICE.md) for Apache-2.0/BSD-2-Clause notices and limitations.
