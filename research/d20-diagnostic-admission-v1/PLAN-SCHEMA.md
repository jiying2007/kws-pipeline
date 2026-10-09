# D20 no-model preparation: plan shape only, execution NOT_READY

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
child. Reads are capped at 64KiB, output at 16KiB; versions are character/length-limited.
Only structured capability counts/limits are emitted; raw cgroup paths, mount
lines and child stderr never appear. Child failures use fixed error codes.
It ALWAYS reports execution_ready=false. It neither configures nor proves
a hard resource limit, backend thread control, dependency identity or admission.

`contract.py` is a PLAN-SHAPE CHECKER ONLY. The function `validate(plan)` checks
798 distinct row/stage/backend labels, shapes, byte counts, syntactic digests,
paired digest equality, frozen expected source-hash strings, original gates and
versions, a single declared C-saved anchor, and no output chaining. Its return
always says numerical_admission=false, execution_ready=false and
input_authenticity_verified=false, even for a syntactically valid plan.

It does NOT open source files, verify bytes, authenticate declared provenance,
bind a tensor to an NPZ array key/frame, verify CMVN provenance, bind slices to
the real weights, or establish C/Torch dispatch identity. Matching hash strings
are claims, not verification. Synthetic fixture hashes are placeholders only.
No numerical execution caller may use this checker as its authorization gate.
Unknown identity MUST remain fail-closed at that future gate.

`python -B -m unittest discover -s . -v` tests one synthetic shape-valid plan and
17 invalid metadata mutations, plus four bounded-output/redaction tests. Run also with `-O` and `-OO`: validation uses
explicit exceptions, not removable asserts. These are metadata tests, NOT
single-op tests or numerical backend qualification.

## Current independently observable environment facts

Rechecked on 2026-10-09 with the included checker: Python 3.12.14;
Torch distribution NOT_INSTALLED; NumPy distribution 2.3.5. Required versions
are Torch 2.11.0+cpu and NumPy 1.26.4. Metadata is not an imported-runtime check.
CPU-only standard-library child affinity could be restricted to [0] and it had
one thread. Torch intra/inter-op dispatch/threading remains unverified.
/proc/self/cgroup reports 0::/; no cgroup mount is visible, and
/sys/fs/cgroup/memory.max is absent. There is no verified 512MiB RSS hard-limit
mechanism in this environment. RLIMIT_AS is NOT RSS; RSS sampling is not a hard
limit. No unconstrained Torch import was attempted to measure whether it fits.

## Driver design and missing preconditions

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
3. A real materializer must hash-check archive/member bytes, verify NPZ names,
   array keys, 9+10 frame identities, dimensions/dtype/order, saved CMVN provenance,
   weight offsets and signed-zero bytes. It must bind each planned row to those
   exact bytes and independently verify both backends receive the same data.
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
