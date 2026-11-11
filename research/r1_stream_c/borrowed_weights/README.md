# R1 qhead8/par4 immutable model view

This opt-in resource candidate reuses the exact existing 1,960,140-byte INT8
classifier-head model and the existing `par4` arithmetic. It adds an explicit
borrowed-model constructor to a fresh SDK tree. It does not train, quantize more
layers, change the decoder, or replace the copied constructor.

The existing caller reads a complete model payload, then SDK construction
allocates and copies another complete payload. Both allocations coexist during
construction. `fsmn_sdk_create_view` validates the same bytes and binds the
model directly to the caller allocation. This removes the second model-sized
allocation and copy. A process peak-RSS reduction is a measurement to obtain,
not a value implied by the source or the model's byte count.

## Ownership contract

- The model buffer must be 8-byte aligned, readable for the exact model length,
  and immutable from the beginning of construction through SDK destruction.
- Populate fresh `malloc` or aligned allocation directly with `fread`/`read`,
  or use read-only `mmap` storage. The model is read through typed float/double
  views; a declared C byte array, including `memcpy` from that array into a heap
  allocation, is not supported. `aligned_alloc` capacity must be rounded to an
  alignment multiple (1,960,144 for alignment 8); the API length remains exactly
  1,960,140 bytes. The included runners use `malloc` and `fread`.
- Ownership remains with the caller on every outcome. Release the model only
  after `fsmn_sdk_destroy`. No other thread or process may modify a shared view.
- Frontend tables retain their existing copied semantics and can be released
  after construction. The original `fsmn_sdk_create` still makes a model copy.
- Size, model SHA-256, source identity, package layout, finite-value, CMVN,
  scale, overlap and platform checks remain active. Failed construction does
  not modify or free the caller payload. A misaligned view is rejected.

The borrowed stream explicitly sets its owned-weight pointer to NULL. Its network pointers
refer to the caller model; the unchanged destructor frees only SDK state.
There is no feed-time model hash, allocation, copy or new arithmetic. The
caller must uphold immutability after the initial validation.

## Build without running a model

```sh
python3 -I -S -B research/r1_stream_c/borrowed_weights/materialize.py \
  --repo . --output /fresh/r1-borrowed-sdk
cmake -S /fresh/r1-borrowed-sdk -B /fresh/r1-borrowed-build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build /fresh/r1-borrowed-build --parallel 2
```

The generator authenticates the frozen public archive, the existing quantized
materializer and every source file it transforms. It stages changes before
publishing a fresh output directory. It does not modify the original source,
model, default SDK, saved observations or existing constructor implementation.
An optional `--model` plus `--frontend` reuses the existing exact-model packer;
all generated weight files remain owner-held research assets.

`fsmn_run` retains its copied-weight behavior. `fsmn_run_view` uses the explicit
borrowed API and accepts the same research-only PCM/WAV command line.
`fsmn_trace_view MODEL FRONTEND MONO16K.PCM` emits the original F/E trace format
on stdout. Its stderr receipt adds model-construction wall milliseconds and
Linux process peak RSS, with `physical_ssc305:false`. Construction timing
excludes file reading; peak RSS covers the complete process, including input
loading and tracing. These are host measurements and not live-AFE latency.

`fsmn_init_bench (copy|view|mmap) MODEL FRONTEND` performs **one**
initialization and destruction with zero PCM and zero model forwards. The
original `copy` and `view` modes remain unchanged. The **opt-in `mmap`**
mode opens the exact-sized regular model file with `O_NOFOLLOW|O_CLOEXEC`,
maps it `MAP_PRIVATE|PROT_READ`, passes its file-backed pages directly to
`fsmn_sdk_create_view`, then destroys the SDK **before** `munmap`.
No caller heap allocation or `fread` copy of the 1,960,140-byte model is
required. Frontend tables are still copied inside the SDK. The model's
original SHA-256, layout and finite-value validation is unchanged. Symlinks,
special files, truncated/wrong-size models and failures are rejected.

**Immutability is a caller/environment requirement.** The underlying model
file must not be modified/truncated/replaced *in place* while mapped or
referenced by the SDK; an immutable squashfs file is an appropriate example.
A read-only memory mapping does not protect against an external writer
mutating or truncating a writable underlying file. Never use a mutable
temporary model path as proof of safety.

The constructor receipt reports `model_init_wall_ms` and also
`model_load_and_init_wall_ms` / `model_load_and_init_cpu_ms` (starting
before model file access). Those metrics must not be conflated: `mmap`
defers file-backed page faults until SDK validation touches the model.
Keep file-cache state, input SHA256, memory pressure and frequencies comparable
when screening initialization CPU. Linux `/proc/self/status` VmHWM/VmRSS
and optional `/proc/self/smaps_rollup` PSS/Private_Dirty support a more honest
memory comparison than resident bytes alone. Unavailable `/proc` fields
remain null. The `getrusage` peak RSS and minor/major faults are reported
separately. Process peak RSS includes the full process, and may not decrease
even if anonymous heap usage decreases. **No streaming CPU reduction is
implied by this change.**

Example source-only smoke on invented model bytes (real model validation must
reject them; source/mock tests use separate explicit stubs):

```sh
# Build the isolated borrowed SDK as documented above, then use owner-held
# verified model/frontend paths OUTSIDE the public repo for actual measurements.
fsmn_init_bench copy /private/model.r1hq8 /private/frontend.f32le
fsmn_init_bench view /private/model.r1hq8 /private/frontend.f32le
fsmn_init_bench mmap /private/model.r1hq8 /private/frontend.f32le
```

Rotate modes across independent low-RSS processes, save unmodified receipts
and compare load+init CPU, VmHWM, RSS and PSS. Neither source-only CI nor host
init measurements certify real SSC305/BSP compatibility, long-run accuracy,
FAR/FRR, AFE latency, power or thermal behavior.

## Optional mmap for actual offline streaming

The existing [source-bound target resource benchmark](../quant_head8/target_resource_bench.c)
now accepts **`--mmap`** only when compiled against a generated borrowed-view
SDK header and its matching library. This is a real file-fed streaming SDK
application: it maps the verified model as read-only, constructs the existing
view SDK, feeds genuine owner-held PCM, measures the complete stream CPU/RTF,
then destroys the SDK before unmapping. Original three-argument (copied
weight) behavior is unchanged.

Use the compiler, `-I` SDK include path and static library from the same
materialized SDK. Run a strictly identical PCM and target CPU frequency
against the existing copy-mode control. The generated SDK already exposes
`fsmn_sdk_create_view`; no extra inference engine, allocator or model format
is introduced. See [target resource guide](../quant_head8/TARGET_RESOURCE_PROFILE.md)
for the exact command and the separate device/acoustic qualification limits.
The frozen historical SDK header explicitly rejects `--mmap` without a
borrowed-view API instead of silently falling back to model copying.

## Bounded evaluation and tests

The completed 2026-11-11 run is recorded in
[OBSERVED_20261111.json](OBSERVED_20261111.json), with a separately recomputed
[independent audit](INDEPENDENT_AUDIT_20261111.json) and a
[Chinese result report](../../../docs/KWS_V3_RESOURCE_20261111.md).
Both host and ARM/QEMU candidates matched every saved F/E trace. The measured
37.58% reduction applies to host initialization VmHWM; streaming steady-state
RSS was not measured, and post-construction RSS was essentially unchanged.

Use the saved exact 134-record input manifest and compare each new trace with
its retained `arm_par4` trace. Count the same 12,534 source frames and 88 keyword
proposals; replay does not add independent samples. Measure copied versus
borrowed initialization in independently launched low-RSS processes without
feeding PCM. A new build's preparation receipt always says NOT_RUN until
actual, separately retained evidence exists.

Source-only tests exercise the real loader using clearly invented structural
fixtures, and the constructor ownership paths using explicit link-time stubs.
They make zero model forwards and cannot grant acoustic or hardware approval.
Run `python3 -I -S -B research/r1_stream_c/borrowed_weights/test_contracts.py
--cc gcc --sanitize`; repeat `--cc` for another required compiler. The tests
also compile all real runners, require rejection of an invented model, and
exercise failed destruction and `/dev/full` receipt writes with explicit SDK
stubs. Constructor tests balance every model and SDK/core allocation and free.
`--sanitize` enables ASan/UBSan with leak detection disabled; opt in separately
with `--leak-check` only on an executor that supports LeakSanitizer. A failed
LeakSanitizer attempt must be retained, never described as a passing leak test.
The original FP32-vs-Torch gate remains FAIL_UNCHANGED; independent human FRR,
continuous natural FAR, calibrated decisions, final-AFE latency and physical
SSC305 resource/soak qualification require their own evidence.
