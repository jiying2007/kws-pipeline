# Isolated C0/C1 streaming encoder source

This is **isolated source and bounded numeric-contract preparation**, not a
trained candidate or admission to a training/quality experiment. The dated [parallel comparison plan](../../configs/research/parallel-candidate-screen-2026-10-10.json)
remains disabled and unchanged. Training, Torch/C or integer parity, trained
weights, continuous-event quality and SSC305 measurements remain **NOT_RUN**.
Numeric API results require a successful exact-source CI receipt; test presence
alone is not a numeric pass.

## What is implemented

- A dependency-free specification and symbolic state/parameter/layout description.
- An explicit-instance adapter for the existing C0 `TinyStreamingRNN`.
- A separate research C1 causal depthwise-temporal encoder source with caller-owned
  per-layer history. It is a local reduced design, not a reproduction of an
  upstream published model or its reported accuracy.
- Standard-library contract, AST and shape-only checks in existing CI. Shape-only
  objects are not numerical tensors and do not establish inference correctness.
- An explicitly authorized, fixed-seed random-feature CPU numeric contract in the
  existing pinned-Torch `objective-math` job. It has no audio or optimizer updates.

The package has no automatic execution on import, training loop, model download,
runtime registration, weights loader or product runner. The separate numeric
test explicitly constructs random models. No product `.kwm` file is generated. Current
`training/model.py`, the product trainer/exporter, C runtime, model registry,
thresholds and shipping policies are unchanged.

## Interface and lifecycle

The specification fixes 32 input features, five existing CTC outputs and zero
future-frame context. C0 uses the existing 64-dimensional recurrent state. C1
requires explicit width and per-layer kernel/dilation choices; there is no
recommended, measured or automatically selected default topology.

The source interface is:

- `init_state` / `reset_state`: return fresh zero history for a specified batch,
  device and dtype; the caller replaces old state on a discontinuity.
- `step`: `[B,F]` features plus explicit state produce `[B,V]` logits and state.
- `forward_chunk`: `[B,T,F]` plus state produce `[T,B,V]` logits and state.
- `forward`: resets state for a complete sequence and returns `[T,B,V]`, matching
  the reference CTC layout. Non-None `lengths` is rejected; pass unpadded streams.
  This research adapter is not a drop-in padded-batch training implementation.

C1 applies input Linear → ReLU, followed by depthwise temporal convolution →
ReLU → pointwise convolution → ReLU → residual blocks, then an output Linear.
All projections/convolutions use biases. There is no BatchNorm or dropout in this
source variant, so it does not inherit an upstream normalization recipe.

For layer `i`, history length is `(kernel_i - 1) * dilation_i`. Cache stores
that layer's **input**, oldest to newest, not its output. Kernel size one has
explicit zero history; `[-0:]` must never retain the whole input. Returned history
uses bounded independent contiguous storage rather than a view retaining an
entire chunk. No implicit detach is performed: the future training runner must
state its cross-chunk gradient/truncation policy.

An empty chunk returns empty logits and preserves state. End-of-file does not
invent future zero frames or flush a detection. Batch/device/dtype/state shapes
must agree; independent streams must not silently share history. Zero future
frames does not mean zero end-to-end delay: frontend windows, buffering, decoder
and callback timing remain part of the eventual measurement.

## C0 identity and lazy dependencies

The C0 builder requires a caller-supplied existing RNN instance. It delegates the
numeric step to that instance instead of copying the RNN equations or importing
`training.model` implicitly. This avoids this package itself changing global
Torch threading/backend settings. The caller remains responsible for deterministic
initialization and the exact reference-source/model identity.

The builder's structural checks do not authenticate an injected instance or grant
training admission. A future comparison must bind its code, initialized state,
data/split, optimizer and seed receipts before running. Torch is a lazy dependency
of explicit construction, not of importing the package or describing metadata.
C1 explicitly initializes float32 CPU parameters using the caller's RNG state;
there is no implicit seed. Move the model explicitly before other-device inputs.

## Symbolic costs and research export description

Contract costs describe the specified float32 research layout, including bias,
projection and per-layer history. MAC counts are algebraic operation counts, not
runtime measurements; additions, activation, allocation/copying, frontend and
CTC decoder costs must not disappear into a claimed speedup. A small parameter
file does not prove low memory traffic, low RSS or low p99 callback latency.

Parser/allocation limits are defensive structural limits. They are not SSC305
resource budgets or evidence that a topology fits the device. Batch-scaled cache
and per-stream cache are distinct. Real CPU/latency/power/temperature and the
comparison's approved compute budget remain unknown until independently measured.

The described tensor names, dimensions and state layout are a **research layout
inventory**, not a serializer, quantizer, `.kwm` ABI revision or C runtime loader.
It must never be passed to the existing v2 RNN exporter as if compatible. Float32
source weights must not be described as W8A8 or as verified integer inference.
A future export design needs an explicit version and matching parser/kernel,
quantization, alignment and numerical/event equivalence evidence.

## Bounded numeric contract

`tests/test_candidate_encoder_numeric.py` checks only float32 CPU source behavior.
It runs in the existing `product-training-data-contract.yml` `objective-math` job,
reusing its pinned Torch runtime. The numeric step has a five-minute timeout;
Torch intra/inter-op and BLAS pools use one thread, with deterministic algorithms
and MKLDNN disabled. Missing Torch fails rather than silently skipping the suite.

The fixed fixture uses model seed 20261011, feature seed 20261012 and initial-state
seed 20261013; batch 2, 13 frames, and chunks `(2, 0, 1, 4, 0, 6)`. C0 keeps the
required 32×64×5 reference. C1 uses one toy width-8 topology, kernels `(3, 1, 2)`
and dilations `(1, 3, 2)`, exercising a zero-history kernel and causal dilation.
These are test fixtures, not selected candidate dimensions or a search/sweep.

The checks cover:

- C0 reference-source blob identity and direct reference output/gradient parity
- Whole-sequence, frame-step and uneven chunks from zero and nonzero state
- C1 independent literal temporal-tap oracle for convolution order and gradients
- Empty chunks, fresh reset, row/session isolation and zero future-frame context
- Finite input, initial-state and parameter gradients, equal across layouts;
  cross-chunk autograd connectivity without any optimizer or parameter update
- Independent compact returned-state storage, no input/state mutation, and the
  real zero-history storage/gradient case
- Rejected malformed state, dtype, input shape and padded-length arguments

Float32 comparisons use absolute/relative tolerances 2e-6/2e-5 for outputs/state
and 2e-5/1e-4 for gradients. This is tolerance-based equality, not byte identity.
A public numeric-only JSON receipt records source hashes, exact CI identity,
versions, seeds, dimensions, check counts, errors and observed differences. CI
retains the receipt as `candidate-encoder-numeric-<run_id>-<attempt>` for 14 days.
Until that receipt is inspected, numerical execution remains **NOT_RUN**.

A stdlib scope selector uses the full PR merge delta or complete push delta.
Candidate-only source/test/doc changes run this bounded numeric check without
starting the existing product data-download or reproducibility-training jobs.
Any original product-trigger path in the same change preserves those original
jobs; shared workflow/selector changes exercise the original gates as well.
No new research workflow is introduced and the disabled comparison plan is not
changed. Static CI scope tests do not substitute for the numeric receipt.

## Verification boundary and next step

Metadata/AST/fake-shape checks remain separate from the real Torch numeric suite.
A successful numeric receipt would establish only the tested source/API contracts
at this one fixed fixture and tolerance, not accuracy, training convergence,
resource feasibility, C/export parity or product qualification.

Before any trained candidate comparison:

1. Freeze an exact topology and fair sample-exposure/compute budget after review.
2. Supply a reviewed, rights-cleared training base with actual-label/PCM lineage
   and independent train/calibration/selection groups.
3. Implement/version export and verify the actual C path; add further numerical
   coverage for that design. Source-only numeric tests cannot certify it.
4. Follow the existing paired comparison, calibration and fresh final-qualification
   policies. No candidate winner, model-selection seed consumption or product
   approval is recorded here.

中文摘要：本目录增加了隔离的 C0/C1 随机特征数值合同测试，固定种子、单线程、
五分钟上限，不读语音、不下载权重、不做优化器更新。数值通过必须以实际 CI 回执
为准；源码或静态检查存在不代表执行通过。候选代码单独变更不会额外触发训练，
原有训练/数据路径变更仍保留既有检查。数值合同不代表唤醒效果、C 导出或 SSC305 资格。
