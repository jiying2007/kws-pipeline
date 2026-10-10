# D20 no-model preparation: saved-byte binding, execution NOT_READY

This work is based on public fa523b3 diagnostic-readiness and saved-diagnostics
protocols. D20 original FAIL and D90 NOT_RUN remain unchanged. No model/operator,
decoder replay, training, TTS or ASR calls. No numerical backend imports or
large dependency downloads. The original 798 rows / 120s / RSS512MiB limits
and raw/probability/frontend/composed-CMVN tolerances are unchanged.

## What these tools actually do

`python -I -S -B admission.py --output NEW.json` writes a standard-library-only
read-only environment inventory to an exclusively created file. It inspects
installed distribution metadata without importing numerical packages, lists
visible cgroup information, and verifies CPU affinity in a tiny standard-library
child. Reads are capped at 64KiB (one extra sentinel byte detects overflow), output
at 16KiB; versions are character/length-limited. METADATA is opened as a regular
file with no-follow directory-relative opens, then decoded strictly as UTF-8.
Oversized files, duplicate installations/identity headers, unsupported metadata
layouts, symlinks, invalid names/versions and malformed encoding fail closed.
Neither Distribution.version nor its unbounded metadata/read_text APIs are used.
The no-checkout hosted probe embeds the same reader; CI checks source equivalence
and runs the same positive/negative metadata fixtures against both implementations.
Only structured capability counts/limits are emitted; raw cgroup paths, mount
lines and child stderr never appear. Child failures use fixed error codes.
It ALWAYS reports execution_ready=false. It neither configures nor proves
a hard resource limit, backend thread control, dependency identity or admission.

`contract.py` is a PLAN-SHAPE CHECKER ONLY. The function `validate(plan)` checks
798 distinct row/stage/backend labels, shapes, byte counts, syntactic digests,
paired digest equality, frozen expected source-hash strings, original gates and
versions, a single declared C-saved anchor, and no output chaining. Dimensions,
byte counts and frozen budgets must be JSON integers; equal-valued floats and
booleans are rejected, as they already are for row and stage indices. Its return
always says numerical_admission=false, execution_ready=false and
input_authenticity_verified=false, even for a syntactically valid plan.

It does NOT open source files, verify bytes, authenticate declared provenance,
bind a tensor to an NPZ array key/frame, verify CMVN provenance, bind slices to
the real weights, or establish C/Torch dispatch identity. Matching hash strings
are claims, not verification. Synthetic fixture hashes are placeholders only.
No numerical execution caller may use this checker as its authorization gate.
Unknown identity MUST remain fail-closed at that future gate.

### Saved-input materializer (byte-only)

`saved_inputs.py` closes the source-byte link that `contract.py` intentionally
does not provide. It accepts exactly the four frozen files listed in
`PUBLIC-READINESS-SUMMARY.json`, with their full SHA-256 identities and byte
lengths, under a caller-supplied local root. There is no hash/layout override,
download, backend import, shared-library load, execution mode or runner.

```sh
python3 -S -B research/d20-diagnostic-admission-v1/saved_inputs.py \
  --saved-root /path/to/already-restored-saved-files \
  --output /path/to/new-private-binding-directory
```

All four full-file hashes must pass before any archive, NPY or JSON parsing.
The fixed NPZs contain 52 Torch and 58 C members, using DEFLATE and NPY 1.0 with
128-byte data offsets. Every member name, shape, dtype, order and byte length is
checked, and member/payload digests are recorded. The bounded NPY-header pattern
extends `cosyvoice3_cross_voice/packager.py`; historical readers are unchanged.
No floating-point values are unpacked, rounded or recomputed.

- Global rows 0–8 use part 0; rows 9–18 use part 1. Both saved `counts` arrays
  must be exactly the little-endian int64 bytes for `[9, 10]`.
- Stage 0 uses the C-saved `cmvn{part}` output, never raw `input` or
  `cmvn_same{part}`. This is a separately saved call with the same C input and
  CMVN function, not an internal stage-0-input hook. The report explicitly sets
  `historical_stage0_input_hook_recorded=false` and
  `cmvn_observation=saved_separate_call_same_c_function_and_input`.
  Stage `s > 0` uses C-saved `stage{part}_{s-1}`.
- Memory stages 4/8/12/16 share the same actual C `cache_before_{part}` row,
  all 22,528 bytes in `[128,11,4]` order. The layer index and before-row-update
  state are explicit. No cache is reconstructed or shifted by the materializer.
- The exact export's 30 tensors are checked in fixed contiguous byte-offset
  order, including each tensor SHA. Only the stage's own weight names are bound.
  Stage 0 does not apply CMVN again; its weights are the first affine's weights.
  The research export has six symbols `<blank>/你/好/小/窝/屋` and output width 6;
  production transcript-mapping rules are not applied to this historical export.
- `inputs.bin` starts with the unchanged exported weight payload, followed by
  one cache slice per row and the saved stage-input slices. `binding.json`
  records exact source/member and bundle offsets, the existing 798-job plan,
  hashes and remaining blockers. The C/Torch jobs reference equal byte ranges;
  every pair is dereferenced and byte-compared before output. Both written files
  are re-opened and hash/length-checked before a success receipt is printed.

The output directory must not already exist and is created with private access.
Do not commit or upload these generated weight/input bytes. The stdout receipt
contains only identity metadata. It reports `saved_byte_identity_verified=true`
only after the fixed source checks, while `backend_dispatch_identity_verified`,
`numerical_admission` and `execution_ready` remain false. This proves saved-byte
mapping, not actual backend consumption, numerical correctness, historical build
identity, CMVN arithmetic correctness, or equivalence to historical chunk dispatch.
A future runner must re-authenticate its source and consumed buffers; a copied or
edited JSON receipt is not an admission token. `contract.validate` still returns
`input_authenticity_verified=false` for every plan, including this generated plan.

```sh
python3 -S -B research/d20-diagnostic-admission-v1/test_saved_inputs.py
python3 -S -B -O research/d20-diagnostic-admission-v1/test_saved_inputs.py
python3 -S -B -OO research/d20-diagnostic-admission-v1/test_saved_inputs.py
```

These ten tests cover only small valid-format byte/metadata fixtures, all row
and stage mappings, exact offsets, signed-zero bytes, and shared pair slices.
Two ordinary-file expected SHA/length mismatches fail in the byte reader. A mocked
fourth-file authentication failure through the public CLI additionally verifies
that all four reads precede archive/JSON parsing, plan construction, output
directory/file creation and success receipts. No archive codec is exercised by
this failure test.
Synthetic helper results cannot claim authenticity. Tests use no model, archive
codec/decompression counterexamples, backend, resource qualification or runtime
probe. CI runs the same tests in the existing standard-library source-check job.

The CMVN provenance check used the hash-verified saved worker
`local-certificate-v2-failed/qualification/src/worker.py` (11,053 bytes,
SHA-256 `dfb089e16122ea7d6bd9c5cd7336f6096397e60e15f431c72d514814ec85867b`)
from the [fixed public archive catalog](https://github.com/jiying2007/kws-data/blob/d9a65cc616cbb0e55a99f4c0e77c16e29bc6622a/research/2026-10-08-d20-d90-host-research/CATALOG.json).
Its `native_sequence` saves `bridge.cmvn(x)` separately, while `cmvn_same` uses
the reference input. The [historical precision64 C source](https://github.com/jiying2007/kws-pipeline/blob/0b748959a414fb3a02ef3dfec1de1cc5278a40a9/research/native_a20/baseline/precision64/model/a20_fsmn.c#L25-L47)
(3,193 bytes, SHA-256
`40ddbb89ddea5bbd7b989046279a7bc56688b922563ddb0969fa1dc9fbe69a68`)
uses the same `a20_cmvn` function inside `a20_step` before stage 0. This establishes
the saved key's static provenance; it does not prove a new build, floating-point
environment or dispatch equivalence. Neither historical source was executed.

On 2026-10-10, the real four-file saved-only materialization passed all file,
member and tensor checks and produced 798 planned jobs. The output binary was
2,312,816 bytes, SHA-256
`c26c0d3a675623f6c4ec9ad3d243c765a3614876072370e65d8d16286b7f1392`;
the binding JSON was 877,794 bytes, SHA-256
`bd86b3d6d0ccfa23f8389099fdafc55b507c4f222ee9f4196a8fbb26fec728fc`.
Both files were re-opened and verified. Outputs were kept outside the repository.
An independent standard-library readback, without importing the materializer's
helpers, matched all 798 job-input slices, 152 job-cache slices and 30 weight
offsets directly against the original saved bytes.
All nine focused tests passed in normal, `-O` and `-OO` modes. No model/operator
calls or numerical imports occurred; execution remained NOT_READY.

`python -B -m unittest discover -s . -v` tests one synthetic shape-valid plan and
17 invalid plan mutations and JSON integer-type regressions, plus bounded-output/redaction and real temporary
METADATA fixtures (including the 64KiB boundary). Run also with `-O` and `-OO`: validation uses
explicit exceptions, not removable asserts. These are metadata tests, NOT
single-op tests or numerical backend qualification.

## Historical local environment observation (2026-10-09)

Observed on 2026-10-09 (a dated snapshot, not a live environment claim): Python 3.12.14;
Torch distribution NOT_INSTALLED; NumPy distribution 2.3.5. Required versions
are Torch 2.11.0+cpu and NumPy 1.26.4. Metadata is not an imported-runtime check.
CPU-only standard-library child affinity could be restricted to [0] and it had
one thread. Torch intra/inter-op dispatch/threading remains unverified.
/proc/self/cgroup reports 0::/; no cgroup mount is visible, and
/sys/fs/cgroup/memory.max is absent. There is no verified 512MiB RSS hard-limit
mechanism in this environment. RLIMIT_AS is NOT RSS; RSS sampling is not a hard
limit. No unconstrained Torch import was attempted to measure whether it fits.

## Current admission checklist

This is the single current diagnostic preparation checklist. The [PR #502
supplement](../diagnostic-readiness-2026-10-09/PUBLIC-PROTOCOL.zh-CN.md) and earlier
protocols preserve dated observations; their older preparation order is superseded.
Prioritize existing saved traces over new acquisition. Matching PCM across runs
still does not establish identical initial state or resolve the original FA cause.

Before using the saved PR450 pointer, run these offline metadata regressions:

```sh
python3 -B tools/verify_durable_trace.py
python3 -B research/diagnostic-readiness-2026-10-09/test_durable_trace.py
```

They bind the fixed source, original ZIP, distinct wrapper ZIP, three ordered
parts and their byte sum, four verification-input identities, and all 44 existing
member identities. They read neither archive payloads nor network resources and
do not claim a fresh restoration/hash verification of payload bytes. Full recovery
remains the fixed source verifier's separate saved-byte-only task. No historical
scripts, model calls or replay are authorized by either metadata test.

D20 driver status: NOT_READY. No driver or numerical runner is published here.

1. Bind the historical C source, compiler flags, FP/FMA environment and reference
   primitive source to the frozen historical artifacts. Public native and
   precision64 variants both exist; choosing one without historical build
   evidence is insufficient. No broader historical source extraction is needed
   or authorized by this preparation.
2. Implement and independently review exact one-stage wrappers. Stage 0 must
   consume authentic saved CMVN OUTPUT, not raw frontend input; other stages
   consume the corresponding saved pre-input. Stages 4/8/12/16 use the identical
   C-observed [128,11,4] saved cache for both backends. Torch reconstructed cache
   remains labelled reconstruction, not observation. Single-row dispatch does
   not establish equivalence to historical chunk dispatch. Never call a full
   model or feed fresh outputs to subsequent stages.
3. Use the byte-only `saved_inputs.py` materializer above to authenticate and bind
   the fixed NPZ members, 9+10 frame identities, saved CMVN/pre-input/cache bytes,
   exact export offsets and byte-preserving slices. Independently review this
   mapping and its source provenance. A future numerical wrapper must separately
   verify what both backends actually receive; planned shared bytes do not prove
   dispatch or consumption. The materializer does not implement such wrappers.
4. Establish a verified dedicated hard-memory scope, ancestry, swap/OOM policy,
   CPU/backend single-thread behavior, no GPU/network and one execution process.
   A stricter cgroup charged-memory limit may be acceptable only if accurately
   described and independently verified, not relabelled RSS measurement.
   Under the scope, verify exact dependencies and import-only peak before any
   numerical calls. Do not install/import unconstrained to discover whether it fits.
5. Only after separate numerical execution authorization and all checks pass,
   one process may reserve at most 798 evaluations with a 120s watchdog and
   RSS512MiB hard limit. Count before each call; abort on any identity, resource
   or contract failure. Never change budgets or retry until success.

No PASS result here authorizes numerical execution or overwrites original FAIL.


## Free GitHub-hosted candidate (not dispatched)

The current checkout already has `.github/workflows/github-hosted-resource-probe.yml`:
standard `ubuntu-24.04`, permissions {}, isolated stdlib `-I -S`, no checkout,
package installation, model imports or tool-driven network requests. Reuse its
allowlisted observations before inventing another broad resource probe. It only
reads visible root cgroup files, not effective process ancestry/enforcement.
Existing CI also tests that probe contract. No workflow was dispatched here.

GitHub documents standard runners for public repositories as free, and Linux
VMs as supporting passwordless sudo. This makes a separate free Linux VM a
plausible candidate; it does not establish delegated cgroup availability or a
safe hard-limit implementation. The local environment blocker is LOCAL ONLY.
Source: https://docs.github.com/en/actions/reference/runners/github-hosted-runners

For this change, CI may run only the synthetic tests in one existing stdlib job;
do not add a new inventory execution step or print its output to CI logs. Actual
hosted observations should reuse the existing resource-probe workflow.
The inventory does not fail CI merely because execution_ready=false; successful
observation or metadata tests do not imply admission. No backend imports, security
changes, package downloads or model calls are part of these checks. The
120s/512MiB numerical contract is NOT claimed enforced by a CI timeout. A true
512MiB hard scope must be established before even the exact-backend import test.

Future isolated import-only admission needs a separately reviewed, ephemeral VM
scope: pre-stage hash-verified public wheels outside the measured scope; check
architecture/Python tags and exact wheel identity; remove network/GPU access
inside the test scope; establish a dedicated cgroup with memory.max<=536870912,
memory.swap.max=0, a verified process membership and OOM policy, then drop
privileges so the child cannot escape or alter its limits. Observe charged-memory
peak and process RSS separately; establish strict limit semantics with a small
synthetic allocation test before Torch import. Bound the import phase by 120s;
pin one CPU and require one process plus backend intra/inter-op threading checks.
If this cannot be done without changing security-sensitive settings, obtain
specific approval for that scope before configuring it. No sudo, cgroup writes,
network namespace configuration, wheel installation or Torch import was performed
in this preparation. Existing public-runner availability and free billing should
be confirmed on the selected job before future dispatch; paid/larger runners
are excluded. Exact dependencies/source binding and isolated import feasibility
remain unknown there until measured; execution_ready remains false.
