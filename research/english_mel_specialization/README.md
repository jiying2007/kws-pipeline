# Exact mel512 C implementation verification

This directory contains one independent implementation experiment for the fixed
English shared feature backbone. It specializes only the two existing mel
Conv1D nodes. The baseline implementation, its source and pins, the original
backbone protocol, training selection and all model assets remain frozen.

The candidate has not been run on the real fixed graphs by the source tests.
The integrated pins enable exactly the first public-main successor of
`cd3e07ebc25505648d10166a806a05afade37703`, attempt 1, with the reviewed worker
and candidate hashes. PR checks do not execute the real fixed graphs. A successful source check is not a
numerical, performance, KWS or SSC305 result.

## One implementation change

`fixed_frontend_mel.c` is a separate translation unit retaining the original
`fixed_frontend.h` and every `ef_*` interface. Its specialized branch requires
exactly Conv, rank 3, batch 1, one input channel, 257 output channels,
`[1,1,16000] × [257,1,512] -> [1,257,97]`, stride 160, dilation 1, group 1,
zero padding, no bias and the original supported attributes. Other operations
use the retained generic implementation.

For this exact branch the addresses are `a[160*frame+tap]`,
`weights[512*channel+tap]` and `out[97*channel+frame]`. The last input index is
15,871, so the original bounds checks are always true. The candidate keeps the
channel, frame and tap order; each output starts with float zero and accumulates
the same ordered products. It does not use FFT, reassociation, SIMD intrinsics,
`restrict`, additional history, new normalization or different model weights.
Compilation keeps `-O2 -fno-fast-math -ffp-contract=off` and the original warnings.

The actual graph manifest contains no 1×1 Conv. The two mel convolutions account
for 25,527,296 of 66,666,624 executed Conv/MatMul MAC per feature window,
approximately 38.29%. That is a static operation count, not a measured time share
or a speedup prediction. The candidate keeps workspace and state sizes unchanged.

## Fixed prior evidence

| Identity | Frozen value |
|---|---|
| Repository | `jiying2007/kws-pipeline` |
| Completed baseline run | `43829765872`, attempt 1 |
| Baseline main | `cd3e07ebc25505648d10166a806a05afade37703` |
| Artifact | `14208696183`, `english-backbone-parity-43829765872` |
| ZIP size | 27,306,350 bytes |
| ZIP SHA256 | `dfe28da9c5a35fef3768f7958ff7945e587645554d838dcbb4aa1d0508a9db1f` |
| Raw baseline RESULT SHA256 | `75d48edbd790f17c55c887b4fb4af9980bdeea3f3925530183822eac89f63737` |
| Original backbone PLAN SHA256 | `5657e4387909bf7b47ba4ed7adec723005a8428357bd75d3515c3ddfe26e550b` |
| This independent PLAN SHA256 | `cc5cd38e2aa96b37672c10d61b8db2ac2364f3d794a0d9eb5f0afe961a2622aa` |

The worker verifies the original run and artifact API identities, exact ZIP and
raw RESULT bytes, all 61 ZIP members, and the closed 49-file result manifest.
It reads the original three `input.pcm` files, three `reference.bin` files and six
same-backend old C traces. It also consumes the original `fixed_models.c`,
`fixed_models.h` and `export_manifest.json` bytes. It does not load or execute the
old binaries or rerun the exporter.

The frozen old backbone Python source is imported only for its standard-library
trace parser, numerical comparison and process-summary parser. Its source SHA
and original PLAN are checked before import/use. None of its reference,
ONNX, synthetic execution, build or execute functions is called. No numerical
Python package is imported or installed by this protocol.

## Independent bounded protocol

The profile is `EXACT_MEL512_C_SPECIALIZATION_SAVED_REFERENCE`.

There are three already saved artificial inputs: zero, the fixed integer 400 Hz
sine cycle, and the fixed LCG signal. Each is 48,000 PCM16 samples at 16 kHz.
The chunk sequence remains `1,1279,257,4096`, with reset between cases, no EOF
padding, the first embedding at sample 16,000 and a 1,280-sample hop. Each process
must produce 26 windows, hence 78 windows for each backend.

The only new graph executions are one candidate native C process and one
candidate ARM/QEMU process for each case: **six new C replay processes in total**.
Each backend is compiled once. There are zero new ORT/reference sessions, zero
new model downloads, zero executions of the old implementation, zero training
updates, and zero TTS, ASR or phone-recognition calls. The compiled program has
no head flag, head source or head weights.

Every new trace must pass all three conditions:

1. Its complete bytes must equal the saved old trace from the **same backend**,
   including stage headers, sample clocks, signed zeros and all float32 values.
   Equal rounded numbers or a successful prior RESULT cannot substitute for
   this raw-byte comparison.
2. Against the saved ORT trace, every finite element must satisfy
   `abs(new - reference) <= atol + rtol * abs(reference)` with the original
   limits below. No bounds are widened after execution.
3. The process summary must retain the exact original sample/window counts,
   no-head fields, workspace and backend-specific frontend state size.

| Stage | Values per window | Absolute tolerance | Relative tolerance |
|---|---:|---:|---:|
| Raw mel | 3,104 | 0.01 | 0.00002 |
| Normalized last 76 mel frames | 2,432 | 0.001 | 0.00002 |
| Embedding | 96 | 0.001 | 0.0002 |

Workspace must remain 528,384 bytes. Frontend state must remain 32,080 bytes for
native and 32,056 bytes for ARM. These are checked against both the frozen
protocol values and the corresponding original process receipt.

Timing is profiling only: whole-process wall and host child CPU, including
startup and trace I/O, with QEMU overhead for ARM. The report preserves original
timings and records historical ratios, but the old and new jobs are not a
matched benchmark. There is no speed threshold or parameter search, and neither
an apparent speedup nor numerical success establishes SSC305 real-time behavior.

## Guard, budgets and failure evidence

The workflow permits real execution only on the public repository's first
`main` push after `cd3e07ebc25505648d10166a806a05afade37703`, attempt 1, with
explicit enabled pins, exact worker/candidate hashes and clean tracked sources.
The Python guard independently checks the same event identity. Pull requests
run source tests only. There is no schedule, manual-dispatch entry or retry.

Each compile has a 180-second limit and each replay a 300-second limit. The
worker has one 1,800-second deadline; the shell allows 31 minutes plus a
30-second termination grace inside a 32-minute step. The actual job's declared
steps total 56 minutes within a 60-minute job, including the always-run five
minute artifact upload. Acquisition is limited to the original run API,
artifact API and ZIP. `GH_TOKEN` exists only in that acquisition step.

The worker reserves each build/replay attempt and writes a durable STARTED
receipt before spawning it. It records separate build attempts/completions,
new C attempts/completions/windows, and zero-new-ORT counters. A timeout or
interruption kills the child process group, writes a partial FAILURE receipt
where possible, and never retries. A missing terminal receipt means unknown
interrupted C progress; it does not mean zero C inference. There is no generic
`model_calls=0` assertion for an actual execution.

Artifacts retain the original evidence ZIP and API responses, consumed
generated C, input PCM and reference traces, old same-backend traces, new
binaries/traces, compiler identities, timings, comparisons, source identities,
attempt journals and any partial failures. Each output directory is fresh;
existing receipts are never overwritten.

## Source-only checks and guarded entry point

The source tests use invented tensors, integer PCM, traces, archives and mocked
processes. They do not read the real model weights or execute actual generated
C. `test_mel_specialization.py` compares the candidate with the original C using
invented tensors; `test_mel_specialization_once.py` checks protocol guards,
binding, trace-byte gates, budgets and failure accounting without compiling.

From the repository root:

```sh
python3 -I -S -B research/english_mel_specialization/test_mel_specialization.py -v
python3 -I -S -B research/english_mel_specialization/test_mel_specialization_once.py -v
python3 -I -S -B research/english_mel_specialization/mel_specialization_once.py
```

The workflow repeats these source checks under normal, `-O` and `-OO` Python.
The executable CLI is reserved for the guarded public job; it requires
`--execute --pins --pipeline-source --baseline-zip --baseline-run-json
--baseline-artifact-json --output`. Do not simulate Actions environment
variables to run it locally.

Preparation checks completed without real graph execution: the 15 candidate
source tests and 21 protocol tests passed in normal, `-O` and `-OO` Python.
The workflow parsed as YAML; its four shell blocks passed `bash -n` and three
inline Python blocks passed AST parsing. A read-only call to the archive reader
also accepted the actual frozen baseline ZIP, including its 49-file manifest,
without starting a subprocess or loading a numerical model package. These
checks establish source and evidence-interface readiness only.

## Delivery limits retained

This is shared-backbone implementation evidence only. It produces no KWS head,
does not establish pronunciation acceptance, independent FAR/FRR, word-end
latency, a board audio chain or SSC305 qualification. All admission, human-gold,
promotion and qualification flags remain false. The existing training route
remains closed because its frozen train-Vosen minimum failed; this implementation
experiment neither changes nor bypasses that decision.
