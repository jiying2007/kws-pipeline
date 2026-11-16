# Fixed English feature frontend: source implementation

This directory supplies the narrow C float32 path for the two already selected
openWakeWord feature assets. It changes no model, recipe, head, event threshold,
or research admission. The Python exporter reads local asset files and emits
static C descriptors and constants; it never downloads or executes a model.

**Current evidence is invented tensor/PCM source testing only.** The two real
graphs have not been exported or run through this C implementation during this
preparation. Actual-node acceptance, actual weights, C/ONNXRuntime numerical
parity, ARM parity, resource use and SSC305 compatibility remain unestablished.
The pure static export has its own one-time public Actions guard described
below. Real numerical comparison belongs to a separate, explicitly bounded
task. Passing these source tests is not L2 or L3 evidence.

## Exact scope and asset identity

| Asset | Original bytes | SHA-256 | Fixed graph input → output |
|---|---:|---|---|
| `melspectrogram.onnx` | 1,087,958 | `ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f` | `input` float32 `[1,16000]` → `output` `[1,1,97,32]` |
| `embedding_model.onnx` | 1,326,578 | `70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f` | `input_1` float32 `[1,76,32,1]` → `conv2d_19` `[1,1,1,96]` |

These are the shared feature models from release `v0.5.1`, upstream source
`368c03716d1e92591906a84949bc477f3a834455`, recorded by actual asset audit run
`43810594841`, artifact `14199923535`, on pipeline commit
`511ccee46dc2fc01483b0d598cfd7fefb286bf31`. The prior successful audit established
ONNXRuntime I/O and operator counts, not each node's attributes or C parity.
The original ONNX pair is 2,414,536 bytes. Exported constant bytes, compiler
output and actual runtime memory must be measured; no smaller weight or memory
claim follows from this implementation. Shared feature asset licensing follows
the recorded Apache-2.0 audit; no pretrained wakeword classifier is included.

The exporter accepts only default-domain opset13, IR7, one float32 input/output,
inline constants and the exact pinned asset bytes. Unused `ai.onnx.ml` opset2
import is accepted because it is present in the audited embedding asset; nodes
in that domain are rejected. There is no custom-op, external-data, subgraph,
function, training, dynamically computed shape, quantized or general ONNX API.

| Allowed node | Implemented contract and explicit limits |
|---|---|
| Conv | NCL/NCHW 1D or 2D, optional bias, groups including depthwise, positive strides/dilations, asymmetric nonnegative pads, `NOTSET` / `VALID` / `SAME_UPPER` / `SAME_LOWER`. Explicit pads with non-`NOTSET` auto padding reject. Weight/group/kernel geometry must agree. |
| MaxPool | 1D/2D, nonnegative explicit or automatic padding, strides/dilations; explicit-padding ceil0/1 and right-padding-only last-window exclusion. All-padding windows reject. Indices output/storage_order1 reject. Deprecated auto_pad combined with ceil1 rejects rather than choosing among historical interpretations. |
| Add, Sub, Mul, Div, Pow, Max | float32 multidirectional broadcasting, maximum rank6; Max has1–8 inputs. Other binary operators have exactly2. No implicit integer arithmetic or conversion. |
| ReduceMax, Log, Clip, LeakyRelu | Opset13 reduction axes attribute/keepdims; finite float32 math; Clip uses optional scalar inputs (not legacy attributes); LeakyRelu honors alpha. Empty data tensors reject. |
| MatMul, Transpose | Row-major float32; vector or batched/broadcast MatMul and a complete transpose permutation. |
| Reshape, Unsqueeze, Cast, Constant | Compile-time shapes and views. Reshape honors opset13 zero-copy dimensions and one inferred `-1`; Unsqueeze consumes constant int64 axes. Cast supports float32 runtime identity and constant float32/int32/int64 conversions. Runtime data casts to other types reject. |

Unknown operators, domains, attributes, unsupported forms and inconsistent
shapes raise a named export error. They are never omitted or replaced with an
identity operation. Finite checks reject invalid input/output math. This is a
fixed-model subset, so explicit rejection is preferable to adding unrelated
operator support. Attribute references are the official ONNX schemas for
[Conv11](https://onnx.ai/onnx/operators/onnx__Conv.html#conv-11),
[MaxPool12](https://onnx.ai/onnx/operators/onnx__MaxPool.html#maxpool-12),
[MatMul13](https://onnx.ai/onnx/operators/onnx__MatMul.html#matmul-13), and
[ReduceMax13](https://onnx.ai/onnx/operators/onnx__ReduceMax.html#reducemax-13).

## C API and ownership

`fixed_frontend.h` is the public contract. `ef_graph_run` executes immutable
exported graph descriptors and float32 constants with caller-owned input and
workspace. The core has no allocation, threads, filesystem access or graph
parser. The exporter computes static shapes and a first-fit liveness plan:
reshape/unsqueeze/identity-cast aliases extend the base tensor's lifetime,
branch inputs stay live through their final consumer, and a node's output
cannot overwrite an input used at that node. Constants are read-only and are
not copied into the activation workspace.

`ef_frontend_workspace_bytes` gives the exact caller workspace for a supplied
pair: float PCM conversion buffer plus normalized mel buffer plus the larger
graph workspace. One graph workspace is reused after normalized mel has been
copied. The separate frontend state contains the PCM16 ring and clocks. The
export manifest records tensor shapes, normalized node parameters, original
attributes, offsets, lifetimes and constant/workspace byte counts. These become
actual-graph measurements only after a successful real export.

`ef_frontend_init/reset/feed` reproduces the selected research feature recipe:

1. Accept exact mono16k PCM16 in arbitrary chunks, without resampling or AFE.
2. At absolute sample16000, and every1280 observed samples after it, convert the
   trailing16000 PCM samples to float32 **in PCM16 scale**, without `/32768`.
3. Run the full fixed mel graph on that already available window. Copy its last
   76×32 values and perform float32 `/10 + 2` in that order.
4. Run the embedding graph and synchronously deliver96 floats and the absolute
   `available_after_sample` to the separately frozen causal head.

There is no startup or EOF padding, extra EOF inference, look-ahead, or claim
that supplying16000 PCM samples makes all16000 acoustically represented. The
existing real-worker geometry preflight still needs to inspect the actual mel
Conv weights, stride, kernel and padding before any acoustic-support claim.
This is the same trailing-window research recipe as
`research/english_final_embedding/causal_embedding.py`; it does not claim
equivalence to a different upstream streaming recipe.

`feed` requires the caller's zero-based first-sample index to match the current
clock. `discontinuity(next_input_sample)` clears acoustic history, preserves or
advances that declared clock, and starts a new full16000-sample observation.
The caller must also clear the head's history/event state with its existing
discontinuity API. A callback or execution failure poisons the frontend until
explicit reset/discontinuity; `consumed` reports exactly the accepted samples.
The embedding and optional trace pointers are valid only during their callback.

Conv and MatMul use sequential float32 multiply/add accumulation with Conv bias
added afterward. Build with **`-fno-fast-math -ffp-contract=off`**. The code
rejects `__FAST_MATH__`; no equivalent-to-ORT numerical claim is made for this
accumulation order or for different `libm` implementations before measurement.

## Export and replay interfaces for the separate bounded task

The first static attempt added one **static export only** job for public
repository `jiying2007/kws-pipeline`, main push with exact `before`
`31238cafe3d6668874d446eae5dfa7c55028fb13`, attempt1. The small
`export_fixed_frontend_once.py` wrapper repeats those checks against the actual
Actions event and clean checkout HEAD. It writes `STARTED.json` before the two
fixed ONNX downloads and pinned `numpy1.26.4` / `onnx1.17.0` parser installation.
It then writes a consumed export marker, exports static C/metadata and records
`RESULT.json`, or `FAILURE.json` on an export error. An always-run finalizer
records an incomplete download/dependency/export failure if no result exists;
the artifact step retains these receipts and any generated static source.
Receipts bind source HEAD, code identities, fixed expected asset hashes and
actual verified assets when available, and record **zero model inference,
training, TTS, ASR and phone calls**. This job does not compile or run C models.
The earlier asset inference audit retains its consumed exact-commit guard
`511ccee46dc2fc01483b0d598cfd7fefb286bf31`; it is not repeated.

### Observed first export failure and finite follow-up

Actual run [43820454034](https://github.com/jiying2007/kws-pipeline/actions/runs/43820454034)
at source `6bf94d542572c5302dcccc449cd999cc3ddafa8f` verified both exact assets,
then stopped with `NONFINITE_CONSTANT`. Artifact `14204202178` is 3,693 bytes,
ZIP SHA256 `0198b8d1ac2b988616368b4276287a3346bebe1ef14c7c9f9df0ed83f6249f1b`.
It retains STARTED, EXPORT_STARTED and FAILED_NO_RETRY; no inference or optimizer
was run. That failure does not identify the original constant node or its uses.

The successor static export is separately guarded by first public main push
with `before=6bf94d542572c5302dcccc449cd999cc3ddafa8f`, attempt1, and a new
`english-fixed-frontend-static-export-clip-bound-fix-v2` receipt schema.
It retains the preceding failure identity in every receipt. Its only added
lowering permits a scalar float32 negative infinity used exclusively as Clip
minimum, or positive infinity used exclusively as Clip maximum. Such a bound
becomes an absent optional input. For this runtime's finite input/intermediate
domain, the default float32 extreme and the corresponding infinite bound have
the same result. See the [ONNX Clip13 definition](https://onnx.ai/onnx/operators/onnx__Clip.html#clip-13).
Both original IEEE bits and all bound-consumer positions are recorded in the
manifest. The dummy compile-time tensor has no emitted use or weight storage.

NaN, nonscalar infinity, reversed-sign bounds, nonfinite weights, graph outputs,
arithmetic/alias uses and any additional consumer remain rejected. No C kernel,
feature asset, numeric comparison tolerance or head policy changes. If actual
graph lowering still fails, GRAPH_FAILURE records node attributes and
initializer shape/type metadata without inference. The new actual result must
establish which constants, if any, meet this rule; source tests alone do not.

Source tests call only the wrapper's pure validator with invented dictionaries;
they do not mutate the environment to impersonate Actions or invoke the guarded
execution stages locally. The wrapper has no execution bypass flag.

The following commands describe the future interface; source tests do not run
them against real assets. The asset directory must already contain the exact
two pinned local files. Export output must be a new directory.

```sh
python export_fixed_frontend.py --assets ASSET_DIR --output FRESH_EXPORT_DIR
```

Pure export emits `fixed_models.c`, `fixed_models.h` and
`export_manifest.json`; the C symbols are `ef_fixed_mel` and
`ef_fixed_embedding`. It imports `onnx` only after file hash verification, uses
protobuf/checker functionality only, and does not import ONNXRuntime or NumPy.

```sh
cc -std=c11 -O2 -fno-fast-math -ffp-contract=off -DEF_WITH_HEAD \
  -I FRONTEND_DIR -I FRESH_EXPORT_DIR -I HEAD_SOURCE_DIR \
  FRONTEND_DIR/fixed_frontend.c FRONTEND_DIR/frontend_replay.c \
  FRESH_EXPORT_DIR/fixed_models.c HEAD_SOURCE_DIR/english_causal_head.c \
  -lm -o frontend_replay
./frontend_replay INPUT.pcm FRESH_TRACE.bin 1,1279,257,4096 SAVED_HEAD.bin
```

`HEAD_SOURCE_DIR` contains the unchanged previously frozen head runtime; the
head blob is its existing `EHKWS01\0` little-endian float32 format. Omit
`-DEF_WITH_HEAD`, the head source and the last CLI argument for frontend-only
replay. The test harness may allocate buffers; the core does not. Replace the
compiler with an explicit target toolchain only in the separately authorized
verification task. ARM/QEMU results still do not establish SSC305 performance.

Trace binary format is deliberately small and exact: eight bytes `EFFTR01\0`,
then repeated little-endian `<uint32 stage, uint32 float_count, uint64 sample>`
and that many float32 values. Each embedding has stage1 raw mel3104, stage2
normalized mel2432, stage3 embedding96, and optionally stage4 eight values:
`ready, event_id, logits[3], probabilities[3]`. Stage4 includes unready head
history frames. Taps are produced within the same execution, without rerunning
the graphs. Output files must not already exist. Malformed PCM, model/interface
errors and write/compute failures return nonzero; partial traces are never a
successful run.

## Source verification and remaining evidence

`test_fixed_frontend.py` creates small invented graphs without an ML package,
compiles C with strict warnings, and checks static views, broadcasting, vector
and batched MatMul, elementary math, rejection paths, liveness and state errors.
Its invented PCM frontend has the same boundary shapes but deliberately uses
strided point samples and maxima, not the real feature model. It checks all
raw/normalized/embedding values against hand-computable expectations, arbitrary
chunking, prefix causality, no padding and head integration with invented
constant weights. Those weights are never described as trained weights.

`test_fixed_frontend_independent.py` separately checks hand-calculated Conv,
group/dilation/asymmetric/SAME padding, negative MaxPool/ceil behavior, broadcast,
transpose/reduce and branch-alias liveness, without sharing the C implementation
as an oracle. Run the suites normally and under `-O`/`-OO`; validation does not
depend on Python `assert` statements.

Actual export acceptance and a frozen, independently bounded numerical check
against the saved ONNX/head assets remain the next evidence. Record stage-wise
and final head errors, clocks/events, native and ARM results, generated-source
and binary identities and real resource sizes. Any predeclared parity failure
remains a failure; do not widen a tolerance from the observed result. Neither
that later software comparison nor this preparation provides microphone/AFE
integration, real-human accuracy or a physical SSC305 result.
