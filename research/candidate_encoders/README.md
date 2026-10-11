# Isolated C0/C1 streaming encoder source

This is **source implementation preparation**, not a trained candidate or an
executable experiment admission. The dated [parallel comparison plan](../../configs/research/parallel-candidate-screen-2026-10-10.json)
remains disabled and unchanged. Numerical Torch/C parity, trained weights,
continuous-event quality and SSC305 measurements are **NOT_RUN** here.

## What is implemented

- A dependency-free specification and symbolic state/parameter/layout description.
- An explicit-instance adapter for the existing C0 `TinyStreamingRNN`.
- A separate research C1 causal depthwise-temporal encoder source with caller-owned
  per-layer history. It is a local reduced design, not a reproduction of an
  upstream published model or its reported accuracy.
- Standard-library contract, AST and shape-only checks in existing CI. Shape-only
  objects are not numerical tensors and do not establish inference correctness.

There is no automatic training, inference, model download, runtime registration,
weights loader or runner. No product `.kwm` file is generated. Current
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

## Verification boundary and next step

Only pure metadata, syntax/AST and fake-shape tests are run for this addition.
They catch specification drift, invalid shapes and unbounded-history mistakes;
they cannot verify numerical convolution ordering, gradient correctness, actual
Torch compatibility, accuracy or performance. Existing canonical product CI
continues to cover the unchanged product path, not these candidate numerics.

Before any candidate experiment:

1. Freeze an exact topology and fair sample-exposure/compute budget after review.
2. Supply a reviewed, rights-cleared training base with actual-label/PCM lineage
   and independent train/calibration/selection groups.
3. Admit explicit numerical tests for reference step/chunk/sequence equivalence,
   reset/discontinuity/tail cases and gradients, then implement/version export and
   verify the actual C path. Do not claim success from these source-only tests.
4. Follow the existing paired comparison, calibration and fresh final-qualification
   policies. No new winner, seed consumption or product approval is recorded here.

中文摘要：本目录把 C0/C1 从候选描述推进到隔离的源码准备。C0 委托调用者显式提供的
既有 RNN；C1 有独立的因果卷积与有界状态源码。当前测试仅涉及结构、语法和形状，
未执行候选数值推理、训练或板测，不代表流式数值等价、真实导出或唤醒效果通过。
