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

`fsmn_init_bench (copy|view) MODEL FRONTEND` uses the two public constructors
in the same binary and performs one initialization and destruction with no PCM
feed. It reports initialization time, `getrusage` peak RSS, and separate Linux
`/proc/self/status` VmHWM/VmRSS values; absent `/proc` values remain null. This
allows the execution report to distinguish inherited process high-water values
from the current executable's memory observations. Launch each measurement as
an independent low-RSS process and preserve its unmodified receipt.

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
--cc gcc --sanitize`; repeat `--cc` for another required compiler. The six tests
also compile all real runners, require rejection of an invented model, and
exercise failed destruction and `/dev/full` receipt writes with explicit SDK
stubs. Constructor tests balance every model and SDK/core allocation and free.
`--sanitize` enables ASan/UBSan with leak detection disabled; opt in separately
with `--leak-check` only on an executor that supports LeakSanitizer. A failed
LeakSanitizer attempt must be retained, never described as a passing leak test.
The original FP32-vs-Torch gate remains FAIL_UNCHANGED; independent human FRR,
continuous natural FAR, calibrated decisions, final-AFE latency and physical
SSC305 resource/soak qualification require their own evidence.
