# Qwen12 one-shot controlled native admission capture

Current source state: all 36 original recognizer slots have consumed their one
attempt. The remaining35-r1 run completed 35 new native captures; the original
Qwen clip-000001 error remains historical and is never retried. All six execution
identities are consumed. Eleven sources have full three-ASR chains, of which six
strictly agree after the frozen punctuation-only normalization. Original-group
candidate upper bounds are K1=0, K2=2, repeat-nihao=3 and suffix-kw2=1. The frozen
training coverage is therefore insufficient: no training, export or candidate
readback ran, and no train root or formal source-resolved admission was created.
No unused recognition allowance or automatic continuation remains. The complete
[derived result summary](results/run-38185393757.json) links the immutable original
evidence; this main-branch summary is not an original capture artifact.

The 12MiB transport was independently verified by zero-model run 38170753261 and
used successfully to retain recovery 3 failure evidence. Models, input clips,
generation parameters, token budget, stop/completion rules, resource thresholds,
host cleanup and ZIP/chunk transport algorithms remain unchanged by the new
identity proposal. The earlier scientific fix is a narrow observational projector for four runtime token fields in Qwen's active
GenerationConfig. Prepared JSON serialization and identity remain unchanged.
The full remaining active configuration must match the frozen prepared snapshot,
except max_length must equal the independently observed input-token length +256.

This new scope consumes exactly twelve pre-existing, public Qwen WAVs from the
pinned `kws-data` commit. It does not regenerate audio, reuse historical Qwen16
outputs as Qwen12 results, or change any historical freeze, gate or evidence.
Only a hash-ordered anonymous twelve-row job and its audio enter recognizers.
Setup and inference see only explicitly frozen public code, locked model/runtime
assets and safe run/head provenance. No human labels, target text, peer output,
GitHub token or host environment is mounted/passed into a worker.

## Terminal result: run 38154869005

The one-shot campaign has been consumed. [Run 38154869005, attempt 1, job
114515750086](https://github.com/jiying2007/kws-pipeline/actions/runs/38154869005/job/114515750086)
executed source `3528118f3cdedf391519a5f8c3365ac15eac16a8` and stopped during
SenseVoice prepare, at load/index 0, with worker `PermissionError` and host
`stopped_no_retry` / `ValueError`. The exception message and traceback were not
retained, so the specific failed operation and root cause remain unestablished.
There are 0 started and 0 completed recognition slots; all 36 slots (12 per
family) are `not_run`. Three-family policy freeze and package-budget preflight
did not complete. No native admission or training result was established.

The failed-run evidence was independently read at immutable [evidence commit
8e05b507e8239ed70191c56ff13216690d6c3049](https://github.com/jiying2007/kws-pipeline/tree/8e05b507e8239ed70191c56ff13216690d6c3049/research/qwen12-native-admission-v1/evidence/native-v1).
All seven chunks and two metadata files reconstructed the 1,597,173-byte ZIP,
SHA-256 `1992682c1a1301a43c06143db0fd034a9ed976252f55bef4c32799fff13dae13`.
Git blob, chunk, archive/member hashes, ZIP CRC and source/run/attempt/job binding
passed; all 323 members including the manifest were accounted for. This is
`PASS_CUSTODY_ONLY`, not successful recognition, admission or acoustic evidence.
The [terminal metadata](evidence/run-38154869005/terminal-status.json) retains
the exact identities, counts and evidence limits. The source ref was subsequently
removed by cleanup as recorded below; the consumed campaign remains consumed.
Preserve the immutable source/run/evidence identities and existing evidence ref.
Do not restore the deleted source ref or rerun this campaign.

### Source-ref retention deviation: 2026-10-11

[Cleanup run 38159168064, job 114528278831](https://github.com/jiying2007/kws-pipeline/actions/runs/38159168064/job/114528278831)
executed main `e9297163a21f8f7c2f960301ff05f1894617d2cf` and deleted only
`research/qwen12-native-admission-v1` at `2026-10-11T17:19:44.6256799Z`.
The workflow considered its tip already contained in main; this one-shot branch
was absent from the retention roster. A later source-ref API read returned 404.
The immutable source commit, original failed run and evidence commit remain
available. Evidence commit `8e05b507e8239ed70191c56ff13216690d6c3049` has the
original source `3528118f3cdedf391519a5f8c3365ac15eac16a8` as its sole parent.
The [dated lifecycle record](evidence/run-38154869005/source-ref-lifecycle-2026-10-11.json)
binds the deletion and subsequent observations without rewriting terminal data.

Cleanup now has a separate exact-name retention guard for the original source,
recovery1 source and consumed synthetic-preflight source before retired,
ancestor or closed-PR deletion eligibility. It does not recreate a missing ref,
reset a consumed campaign, or establish autonomous replay prevention against
arbitrary external deletion/recreation. Any later execution remains a separate
reviewed operation; a missing ref is never proof that a campaign is unused.

## Terminal result: recovery 1, run 38160295526

[Run 38160295526, attempt 1, job 114531695348](https://github.com/jiying2007/kws-pipeline/actions/runs/38160295526/job/114531695348)
executed source `e9297163a21f8f7c2f960301ff05f1894617d2cf`. All 22 ASR setup
commands completed, including the guarded pre-model dependency import. The
scoped FFmpeg policy intercepted exactly one probe; the preflight recorded zero
model construction, weight loading and audio forward calls. The original
source-level compatibility fix therefore passed this actual import check.

The setup container's pause and final CPU close succeeded at 103.355741 controlled
CPU seconds. Its first `rm --force` client wait timed out after about 5.037 seconds.
Final cleanup re-entered pause or inspect and failed; it did not reach a second
remove command. The retained records do not establish whether the daemon had
already removed the container. The cleanup `finally` path could also have
replaced an earlier supervision error, and its gate result was not retained.
Consequently this is `cleanup_unconfirmed`, not proof of successful supervision.

The setup child recorded 133.834176304 seconds. The host's zero combined-setup
accumulator is not an elapsed-time measurement: cleanup failed before that
addition returned. No native prepare, Whisper setup or recognition occurred;
all 36 slots are `not_run`. Across both consumed executions there are zero
started and zero completed recognitions. No training admission has been created.

Immutable [evidence commit d843cc59113c7fd3ac89a0b4be9db71e5950ea20](https://github.com/jiying2007/kws-pipeline/tree/d843cc59113c7fd3ac89a0b4be9db71e5950ea20/research/qwen12-native-admission-v1/evidence/recovery1)
retains the 86,754-byte ZIP, SHA-256
`9b063d2c92ccf88d67678b032d391bf01e0ceb1a626d5cf23a3f11c0fd250d39`.
Its three repository files and seven ZIP members including the manifest passed
independent source/run/job, blob, chunk, member and CRC verification. The
[terminal metadata](evidence/run-38160295526/terminal-status.json) records the
observations and uncertainty without rewriting either execution's evidence.
Both campaign identities are consumed. The cleanup correction disabled native
execution at that historical source. Recovery 2 subsequently ran once and is
also consumed; its terminal outcome is recorded below. Native execution was
disabled at the subsequent transport-source snapshot; the current recovery 3
static release and held execution ref are described below.

## New executable path

`hosted_run.py` uses the existing official Python image/host-capacity/kernel
scope checks. The fresh `setup_adapter.py` is the reviewed source-screen setup
binding copied to this new scope with task workspace/initial-free limits
changed to 10/12GiB; the recovery section records its additional guarded import policy. It retains 124 locked Qwen/Sense package inputs (119 wheels,
five named sdists built once offline), pinned models, exact package metadata and
source-member checks. No resolver/version upgrades are allowed.

Both isolated runtimes are installed before recognition. All three engines first
load/observe in prepare-only processes with zero audio forward calls; their full
policy and budget proof freeze before three fresh decode processes remeasure the
exact same locks/configs. Each family then gets one attempt per clip. Extra loads
count toward setup/host budgets and each family shares a600s prepare+decode
ceiling. Whisper has a different fresh runtime, using the existing bounded_host/pip_acquire/pip_compatibility/phased_setup
transport and the reviewed native Whisper decode implementation. New native
source/model locks and all native control observations freeze before first audio.
Any setup, identity, capacity, decode, native completeness, deadline, output or
cleanup failure stops all new work. The terminal denominator remains 36 with
explicit error/not_run states. Host terminal rows are accounting, not training
admission; immutable resolver review must still pass before a training root exists.

## Limits and precise claims

- Standard public-repository `ubuntu-24.04` only, 70-minute workflow ceiling.
  Before acquisition, host capacity must actually supply cgroupv2, >=4CPU and
  >=14GiB currently available memory; before setup, >=12GiB free task disk.
- Containers: at most4CPU, 12GiB memory, zero swap, 256PIDs; Whisper acquisition
  splits the same envelope between its existing1CPU/16PID broker and3CPU/240PID
  network-none worker. Inference is network-none/nonroot/read-only/no-capability.
- Combined setup wall threshold1800s; each family600s including model loading;
  each clip120s outer host threshold. The capture host3600s signal timer remains
  armed through cleanup, local ZIP construction and repository-chunk preparation;
  publication is a separate bounded poststep. Scheduling and cleanup can overshoot
  wall thresholds; this is not a real-time kernel timer guarantee.
- Cgroup cumulative CPU sampled threshold: setup3600 + Qwen600 + Sense600 +
  Whisper1800 =6600CPU seconds. All controlled cgroups are charged from creation.
  A new minimal gate keeps each cgroup alive before and after its child so the
  host can meter it, then acknowledge pause and capture usage before removal.
  Receipts report actual poll gaps and missing/unconfirmed accounting. The scope
  excludes host work, Docker image acquisition, host ZIP compression, the separate
  publisher and cleanup after pause. No strict all-host, zero-overshoot cumulative CPU claim is made.
- Inference also has RLIMIT_CPU across its main process threads and a focused
  x86-64 seccomp filter blocking fork/vfork/non-thread clone and x32 syscalls.
  clone3 returns ENOSYS to permit ordinary glibc thread fallback. Pure dispatch
  fixtures do not establish real host compatibility; the actual single job must
  pass these checks in its prepare and fresh decode workers.
- Task workspace10GiB is a sampled logical-tree stop threshold, not a filesystem
  quota. Docker image/layers and tmpfs are outside it. Output32MiB is a sampled
  aggregate stop threshold; container per-file fsize remains independently bound.
- Aggregate acquisition reservations: Qwen/Sense4GiB exact response bodies,
  Whisper1.5GiB bidirectional tunnel recv, image128MiB observed-interface reserve,
  audio1MiB exact response bodies. Sum5.626GiB <6GiB. These are deliberately
  different measured scopes; image cancellation does not prove daemon stop and
  this is not a hard total network-wire cap. No failed image attempt proceeds.

## Evidence and cost

No Actions artifact, cache or Git LFS is used. The complete bounded ZIP and its
exact internal member manifest are preserved. Repository transport divides that
ZIP into at most 48 raw chunks of at most 256KiB, under the 12MiB compressed cap. Its small outer manifest binds
the actual repository/run/attempt/job-key/source head, total bytes/SHA256 and
ordered chunk indices/lengths/SHA256. The reader pins the final evidence commit,
verifies independent GitHub run/job metadata, then reassembles and applies the
same complete ZIP/member checks. This is ordinary repository evidence, not an
Actions artifact. Failed or partial native runs retain all available failure,
completion and not-run records; publishing a package does not admit its rows.

A separate trusted-host poststep receives the ephemeral workflow token. No token
is passed to the ASR command, a model worker, a worker mount or retained evidence.
GitHub grants contents:write and actions:read at job scope; the latter verifies
the current numeric job identity. The workflow exposes the token as an
environment value only to the publisher step, after capture or synthetic setup.
The capture host's 3600s signal timer remains armed through local ZIP construction
and repository-chunk preparation and is canceled afterwards. The final host
receipt precedes those packaging steps, so its timing does not cover every later
packaging instruction. The sampled 6600 CPU-second ledger covers controlled
cgroups; host ZIP compression and the separate publisher's CPU and HTTPS traffic
are outside that ledger. Publication has its own 180s supervised deadline and
4-minute step timeout, inside the existing 70-minute job ceiling. It never retries
recognition or changes capture budgets.
Checkout does not persist credentials. The publisher uses only fixed repository
API endpoints and a create-only evidence ref; an existing ref, API conflict or
invalid/missing package stops publication. No retry, force update, main-branch
change or raw error/credential logging is allowed. The poststep can run after a
failed capture step, but cannot manufacture a complete marker for a missing or
invalid package. Abrupt job/runner termination can still prevent publication.

Only public synthetic fixture bytes or the approved native evidence allowlist
are packaged. No weights, audio, private human reviews, raw environment, arbitrary
error text or signed download URLs are collected. Oversize evidence is a failure,
never permission to drop failure, truncation, source or trace records.

## Review and verification

Run `python3 -B test_contract.py`, then the same with `-O` and `-OO`; tests perform
no network, model imports, acquisition or inference. Also run the sibling native
adapter/Whisper-binding/lock, repository transport and publisher mock tests. `prepare_freeze.py`
runs only after all source is integrated in a separate checkout; it hashes the
exact source/role roster and does not enable execution.

The retained five-predecessor static release removed the native condition's
literal `false && ` and set `release.enabled=true` plus its five review bits true.
That fixed identity has now executed once and is consumed; these historical
source bits do not authorize another attempt or a replacement execution identity.
All five consumed source/run/evidence identities remain unchanged. Dispatch requires literal JSON true for
source, budget, independent review, actual complete selected-transport readback,
historical campaign uniqueness, and bounded public transcript publication
approvals. Missing, false, integer or string substitutes fail before Docker,
network or model imports. Only the fixed fresh public campaign-branch push on
its first attempt can run the native job, after contracts pass. Actual workflow
run_number is preserved, not invented or required to be1.

`required_runtime_preconditions` is an exact, ordered obligation roster for every
real run, not approval flags or a declaration that future checks already passed.
A missing, mistyped, reordered or changed roster fails closed at dispatch. The
single real job still checks host capacity/kernel scope, installs both runtimes,
then performs all three guarded prepare-only loads with zero audio forwards.
It freezes the complete policy/configs and successfully dry-packages the actual
prepared runtime/source/tokenizer/inventory closure plus the 12 × 224 full-prefix
trace stress fixture before starting any decode worker. These conditions need
no separate VM or extra preflight load. Each subsequent fresh decode worker must
still pass its own live kernel/CPU guard and frozen-config/identity checks before
its first forward; later workers are never reported as having passed in advance.

Existing same-run `policy-freeze.json`, prepare run/kernel/CPU/config and zero-call
records, `package-budget-preflight.json`, and the host receipt retain these facts.
The sequential host path propagates any prepare/freeze/package failure
before its decode loop; fresh-worker verification remains mandatory. A static
release approval or synthetic fixture cannot replace actual runtime evidence.
The exact collector continues retaining those records. A rejected package check
now retains a small structured failure record with a fixed reason code and
measured compressed/expanded byte and member counts when available; unknown
failures remain failures. No exception text, paths, environment, model load or
evidence omission is introduced.

## Original activation handoff (historical, consumed)

The activation procedure below records the original pre-run requirements.
The source was reviewed, passed exact-head CI and was merged before the one-shot
campaign ran and stopped as recorded above. Its source ref was later deleted
by cleanup as recorded above and must not be restored; the evidence ref must
remain unchanged. This historical procedure does not authorize a second run.
Source review PRs run read-only fixtures and cannot run either contents-write
publication job.

The zero-model repository-chunk preflight has passed actual connector readback:
[run 38152411613, attempt 1, job 114508486242](https://github.com/jiying2007/kws-pipeline/actions/runs/38152411613/job/114508486242)
executed source `d33107c825511b5be994e6e922f31d435f89d808` and created
[evidence commit 46e605153482d7babd7584fdc960c15022be8c12](https://github.com/jiying2007/kws-pipeline/tree/46e605153482d7babd7584fdc960c15022be8c12/research/qwen12-native-admission-v1/evidence/synthetic-v1).
All 34 files (32 chunks and two metadata files) were read at that immutable
commit; Git blob identities, chunk hashes, ZIP members/CRCs, fixed synthetic
seed, independently fetched run/job metadata and checkout identity passed.
The reconstructed ZIP is 8,326,571 bytes with SHA-256
`0e2b26a972a06dee180ac0a4f6cad13166164a64e8c4918c181b75d086e4f466`.
This proves the historical 8MiB repository transport, not the new 12MiB bound. The previous large-log
readback route remains unsuccessful. The used synthetic source/evidence branches
must not be rerun, updated or deleted/recreated.

The real capture can run only from the canonical public repository's fresh
`research/qwen12-native-admission-v1` branch push with `event.created=true`
and `run_attempt=1`, after its contracts job passes. It can capture once and
publish only to `evidence/qwen12-native-admission-v1`, under
`research/qwen12-native-admission-v1/evidence/native-v1`. Both real source and
evidence refs must be absent and are created once; neither is updated or
deleted/recreated. Root must freshly verify source-branch/run history, the
unique campaign claim and the exact activated source head/freeze/retention
before creating the real source branch. Future runtime gates remain mandatory.

The actual complete prepared closure and stress-package check is a same-run
pre-forward requirement, not an externally pre-passed dispatch approval. A
minimal-config fixture alone is not that proof. The host performs it after the
three prepare-only loads and stops before any forward if the package does not
fit; the existing 10GiB workspace checks remain in force.

This separately reviewed activation delta removes only the native workflow's
literal false, adapts source-only contracts to both reviewed source states, sets
the reviewed release flags, updates the current active-workflow identity with an
append-only maintenance receipt, and regenerates `execution-freeze.json` with its
new SHA bound in `execution-release.json`. It also corrects this documentation.
All source/retention checks must pass again for this exact tree. Historical
workflow/source/freeze identities remain unchanged. Source hashes from the
disabled commit do not describe the activated tree. The approved final commit
must be the new branch push head and the real Actions identity in all captures.

Before that push, root must inspect repository branch/run history and record a
single campaign claim: no earlier run may have consumed this twelve-by-three
campaign. Preserve event.created=true and run_attempt=1. Never delete/recreate
the campaign branch or rerun a failed campaign; a failure requires review of its
retained error/not_run evidence. Source-CI run_number is unrelated and is kept
as supplied by Actions. Do not create a training source root until actual native
results have independently passed the unchanged resolver and admission checks.

## Recovery 1 activation history (now consumed)

Original source/run/evidence identities remain immutable; the deleted original
source ref must not be restored. The separately reviewed single continuation used `research/qwen12-native-admission-recovery1`, with
create-only evidence ref `evidence/qwen12-native-admission-recovery1` and directory
`research/qwen12-native-admission-v1/evidence/recovery1`. The reviewed activation source ran once as recorded above. The subsequent cleanup-only
source restored the native literal stop and disabled the release. Recovery 2
also ran once and was disabled before the 12MiB transport repair. Recovery 1's
pre-run review checked the predecessor's zero recognition consumption, absence
of the new refs/run, exact source CI, budget and independent review. Its release
bound the original run, source, evidence commit, archive hash and 0/0/36
accounting; the recovery 2 release later preserved both predecessors. Across those two executions, the same twelve clips
still permit at most 36 recognition attempts; no automatic retry or ref reuse.

A separate source-level compatibility finding is reproducible without a model:
locked FunASR `load_utils.py` calls an external FFmpeg version probe during
import, while the capture process guard rejects non-thread process creation.
The old run retained no exception stack, so this does not retrospectively prove
its unique failing operation. The new narrow `optional_tools.py` policy verifies
the original source SHA, compiled function, caller path/spec and exact arguments;
only that unused external-backend probe follows the library's unavailable branch.
It declares `disabled_by_capture_policy`, preserves installation status as unknown,
and records actual intercepted probes. Other commands and exceptions retain their
original behavior under the unchanged guard. Original dependency bytes and
scientific loader bodies remain unchanged. The policy does not handle SoX:
there is no established SoX import on the selected SenseVoice path.

The existing zero-model import preflight now runs under that process guard and
policy before model acquisition. It remains a setup child with setup accounting,
network and writable runtime; it is not evidence of the later inference
container's read-only/network-none scope. Its additional per-process 600 CPU
second limit is within the unchanged setup and whole-run thresholds. Every fresh
prepare/decode still verifies the original full kernel scope. Stable optional
policy bytes join the native source lock; separate bounded receipts record
actual probe counts. Native and preflight failures retain safe exception type,
errno, fixed stage and bounded source-relative frames, never exception text,
source lines, locals or environment values. Raw setup child stderr can exist
transiently in excluded runtime logs; only safe projections enter public evidence.

The following records the reviewed recovery budget, not a new execution allowance.
Resource accounting is per execution, not a reset of history. The predecessor
measured 454.195425 controlled-cgroup CPU seconds, 708.389337012 capture-host wall
seconds and 700.114547769 combined setup seconds. Recovery retains the same
6600 CPU sampled stop threshold, 3600 host wall timer and 1800 setup threshold;
old measurements plus these new thresholds are 7054.195425, 4308.389337012 and
2500.114547769 seconds respectively, not zero-overshoot aggregate guarantees.
RAM, disk, output and per-family/per-clip limits are unchanged.

Keep acquisition scopes separate. Previously measured Qwen/Sense response bodies
were 3,340,966,447 bytes; one recovery retains the 4,294,967,296-byte body cap
(combined same-scope ceiling 7,635,933,743 bytes). Previously measured Whisper
tunnel receive bytes were 774,060,234; the new cap is 1,610,612,736 (combined
2,384,672,970). Prior audio bodies were 776,208 bytes and the same twelve inputs
add at most 776,208 (combined 1,552,416). Prior image-interface observation was
45,704,322 bytes; the new 134,217,728-byte image reservation is an observation
reservation, not a hard wire-byte bound. The per-run 6GiB mixed-scope acquisition
reservation must not be presented as a two-run total or exact network traffic.
Only standard public GitHub-hosted runners are permitted; no artifact/cache/LFS
storage or paid runner is added. A failed recovery stops and publishes available
bounded failure evidence; it does not silently authorize another execution.


## Cleanup-state correction and zero-model CI smoke

The host binds each successful create to its complete container ID and uses
that identity for subsequent lifecycle commands. An uncertain create without a
verified ID does not authorize removal by name. Final CPU accounting is attempted
at most once; successful removal cannot erase invalid or unverified accounting.
A pause, inspect or CPU-close error still permits bounded cleanup of an already
owned ID. The original supervision error and cleanup error are retained separately
as bounded metadata; no raw Docker error text or listing is published.

Removal is attempted once. If its client result is uncertain, one bounded,
successful exact-ID query must positively report absence before cleanup is
confirmed. A present container, malformed response or daemon error remains a
failure. Finalization consumes this stored state rather than repeating pause,
CPU close or removal. Cleanup operations have a single 15-second best-effort window with
individual calls bounded to five seconds, including when the primary host
deadline has expired. Receipt persistence follows those operations and is included
in the supervision cleanup-phase timing, not a strict 15-second total. This limited
cleanup can extend observed wall time; it
does not extend setup, model or recognition work, waive a deadline, or validate
missing final CPU usage. Failed-phase elapsed time and available gate results
are retained independently of cleanup success.

A read-only `cleanup-lifecycle-smoke` job is prepared for exactly the first
`pull_request` `opened` event, `run_attempt=1`, in the canonical public repository
with same-repository head branch `fix/qwen12-native-cleanup-source-20261011`.
It needs the contracts job and checks out the exact reviewed PR head. The receipt
keeps both that executed head and the original Actions PR-merge SHA. It does not
run on synchronize, reopen, ready-for-review or manual reruns. A failed initial
contracts job leaves this smoke unverified; later code changes cannot reuse an
old head's smoke as proof of the changed head. A further test requires a separate
reviewed decision, not an automatic retry.

The smoke acquires the same pinned official Python image once and runs only a
static Python `pass` through the production gate and owned-container cleanup.
It mounts only the gate helper and empty scoped directories, with no packages,
models, audio, repository credentials or token. It checks the actual CPU, memory,
swap, PID, capability, seccomp and nonroot scope before releasing the child.
No artifact, cache or LFS object is created. Its bounded public receipt is at
most 32KiB, and records success, failure or uncertain cleanup without raw errors.

This CI test adds one image acquisition and a small lifecycle operation on the
standard free public `ubuntu-24.04` runner. The job ceiling is five minutes; the
helper host deadline is 180 seconds, image-client phase 120 seconds and lifecycle
phase 30 seconds. Failed image-client termination may use the existing bounded
20-second cleanup and never proceeds to the lifecycle. Individual ordinary
Docker calls are at most five seconds. The smoke's controlled container uses the
same four-CPU/12GiB/no-swap envelope and a sampled 60-CPU-second stop threshold;
image acquisition, host work and Docker daemon CPU are outside that ledger.
The existing observed image-interface reservation is not a hard network-wire
cap. One independent 15-second cleanup-operation window, followed by receipt
persistence, may exceed a primary deadline;
that overrun remains a failure. That source prepared the test. The subsequent hosted smoke is evidence only
for its executed head and unchanged cleanup components; it does not establish
that a later head or a third ASR execution has passed.


## Observed zero-model cleanup smoke

[Run 38163142188, attempt 1, job 114540172806](https://github.com/jiying2007/kws-pipeline/actions/runs/38163142188/job/114540172806)
completed the one-shot lifecycle on source
`63968444f30d77e712fb74e207fda87119ae3831`, tree
`7ab29f9d564033790741697a8758678bb8a70c59`. The
[exact public receipt](evidence/run-38163142188/cleanup-lifecycle-smoke.json),
SHA-256 `f5c0a179b87ef7cd03294d83aa6c48b83c561052917102546adf70501bacd087`,
binds thirteen source files, the static `pass` command, the actual head and
Actions merge SHA. Independent API/log/source readback passed. Pause, paused
inspect, final CPU close and force removal succeeded; all owned containers were
removed and cleanup verified. Controlled-container CPU through pause was
0.135884 seconds; helper wall was 4.012934521 seconds. Model, audio and package
counts were all zero. These are separate CI observations, not part of either
ASR run's measured resource account and not total CI CPU or traffic.

Recovery 2 keeps the cleanup helper, host cleanup code, CPU gate and CPU ledger
byte-for-byte identical to this executed source. Its changed workflow, release
and campaign constants are separately reviewed identities; this smoke does not
claim to have executed a later recovery 2 head, any model, or old failed
containers. The consumed smoke event is not automatically repeated.

## Recovery 2 terminal result: run 38167052146

[Run 38167052146, attempt 1, job 114551789796](https://github.com/jiying2007/kws-pipeline/actions/runs/38167052146/job/114551789796)
executed source `42b333f8b96c4c9fcca63c9e275d26a4519737c8`, tree
`6e91e77b1e60e19956664da2eab80f0a929e02af`. Both runtime setups and all three
prepare-only loads completed, with zero audio forward calls. The full policy
was written, but package-budget preflight did not complete. No decode worker
was armed: 0 started, 0 completed and all 36 slots not run. All 54 controlled
containers have confirmed cleanup and final CPU accounting in this run.
Recovery 1's earlier uncertain cleanup is not changed by this observation.

The original host retained only `ValueError`, without its message or traceback.
A separate no-model replay rebuilt the policy byte-for-byte from the retained
source and complete prepare closure. The unchanged full-prefix stress produced
10,191,849 compressed bytes, above the former 8,388,608-byte cap. It contained
3,495 payload members plus the internal manifest and 24,521,115 expanded bytes
including that manifest. Omitting the two post-terminal host/outcome records
in a diagnostic copy still produced 10,183,205 compressed bytes. These are
replay measurements, not a recovered original traceback or the exact historical
preflight byte count. They establish a reproducible capacity mismatch without
changing the original records or shortening the stress check.

Immutable [evidence commit 43113646c12fc9d6c112481bab6818c96f2ce789](https://github.com/jiying2007/kws-pipeline/tree/43113646c12fc9d6c112481bab6818c96f2ce789/research/qwen12-native-admission-v1/evidence/recovery2)
retains the 2,656,120-byte terminal ZIP, SHA-256
`140a18c7429b16674b5a1dc8e1f65f4a1be6b0dd4ecb324b0a4c1ccaff11d157`.
All 455 ZIP members including the manifest, source closure, 36-slot census and
actual run/job/source bindings passed independent custody checks in normal,
`-O` and `-OO` modes. The [terminal metadata](evidence/run-38167052146/terminal-status.json)
records the observed failure separately from replay-based diagnosis. This is
custody and preflight evidence; no acoustic admission or training is established.
The consumed recovery 2 source and evidence refs must not be reused or changed.
All three executions still have zero actual recognition attempts in aggregate.

Recovery 2 recorded 477.281264 controlled-cgroup CPU seconds through acknowledged
pause and 759.388826281 capture-host receipt seconds. This wall interval includes
the failed preflight ZIP construction and excludes later final evidence ZIP
preparation/publication. Its ASR response bodies total 3,340,966,447 bytes;
Whisper bidirectional tunnel receive 774,056,598 bytes; image default-interface
observed receive delta 45,710,666 bytes; and audio response bodies 776,208 bytes.
These separate scopes must not be added into an alleged all-network total.

## 12MiB zero-model transport preflight (separate execution)

The separate synthetic source identity is
`research/qwen12-native-transport-preflight-12m-v1`. Its create-only evidence
branch is `evidence/qwen12-native-transport-synthetic-12m-v1`, directory
`research/qwen12-native-admission-v1/evidence/synthetic-12m-v1`, and fixed purpose
is `synthetic12m`. The source ref joins the separate cleanup retention roster.
The old `synthetic` purpose and original fixed seed/size remain supported for
reading prior evidence; neither old preflight ref is created again or updated.

Only the canonical public repository's fresh push with `event.created=true` and
`run_attempt=1` can run this `transport-fixture` job after contracts succeed.
Pull requests have read-only source tests and cannot publish. The fixture uses
multiple fixed-public-seed byte members, each at most 8MiB, to produce a near-12MiB
ZIP and all 48 chunks without models, audio, Docker, package installation or ASR.
No artifact, cache or LFS storage is used. The standard Ubuntu job retains its
10-minute final ceiling; the publisher retains its 180-second deadline and
4-minute step timeout, fixed HTTPS destinations, bounded requests/responses and
create-only evidence ref. This adds bounded synthetic CPU and repository transfer,
not another model-acquisition reservation or native execution allowance.

That transport repair used public main
`42b333f8b96c4c9fcca63c9e275d26a4519737c8` as its base. The original stress fixture and all
scientific/runtime/cleanup behavior remain unchanged. ZIP/chunk limit constants,
explicit synthetic-purpose selection and safe package-failure diagnostics are
the only executable changes. Complete source/tokenizer evidence is never dropped
to meet a size cap. Local replay must pass the unchanged full prepared closure
at 12MiB, reject it at the historical 8MiB bound, and reject new over-limit data.

The transport source was separately reviewed, merged as public main
`8c4d430ce1077aa9eefa8de130450319c64e545e`, and passed fresh-main checks before
its one-shot zero-model run 38170753261 began. Its entire immutable evidence
commit has been independently read back, including all 48 chunks, Git blobs,
archive/member hashes/CRC, fixed seed, and actual run/job/source binding. The
[public custody metadata](evidence/run-38170753261/transport-readback.json) binds
source `8c4d430ce1077aa9eefa8de130450319c64e545e`, job 114562800402 and evidence
commit `4b0c01cbd825802dd5c373dab33332402a6af01f`, whose sole parent is that source.
All 50 files reconstructed the 12,522,365-byte ZIP, SHA-256
`72fa969479341bfddaa712b600325745d35edad3dae18203d5e2819b17b92424`.
Normal, `-O` and `-OO` verification agreed; all 14 negative cases were rejected.
The publisher took 34 seconds within the 39-second fixture job; these intervals
are not additive and do not measure native CPU or total network usage. There
were no models, runtime installs, audio forwards, ASR calls, training or artifacts.
This live result proves bounded synthetic transport, not native readiness. Conflict, unknown write
outcome or failure ends that one preflight without automatic retry. The custody
record's `native_execution_approved=false` and other status fields describe that
historical zero-model run. They remain unchanged and do not describe current
static release approval. Recovery 3 does not rerun this consumed synthetic identity.

## Historical recovery 2 budget proposal

The [resource proposal](recovery2-resource-proposal.json) remains byte-identical
to the disabled, independently reviewed historical budget snapshot. Its false
disabled/pending status fields describe that historical proposal, and are
deliberately not rewritten as execution results or current release approvals.
It kept the first two actuals and one then-future recovery 2 reservation separate
by measurement scope. The following arithmetic is historical planning, not a
new run reservation or the final three-run consumption total. ASR response
bodies were 3,340,966,447 bytes in each old run; with a new 4GiB body cap the
same-scope planning total is 10,976,900,190 bytes. Whisper tunnel receive was
774,060,234 bytes in the first run and the second never entered that stage;
adding the unchanged 1.5GiB cap gives 2,384,672,970 bytes. The two image-interface
observations, 45,704,322 and 45,707,668 bytes, plus the 128MiB observation reserve
sum to 225,629,718 bytes, not a daemon or wire-byte cap. Audio body accounting
uses 776,208 bytes for each old run, with recovery 1 explicitly inferred from
successful source control flow rather than a standalone receipt; a new 1MiB
reservation gives 2,600,992 bytes. These different scopes must not be added into
one download total.

The two paused-cgroup measurements total 557.551166 CPU seconds; adding one
6600-second sampled stop threshold gives planning arithmetic of 7157.551166,
not an unconditional CPU ceiling. Recovery 1 post-pause/cleanup resource use
remains unknown. The old capture-host receipts total 859.683035284 seconds;
adding a new 3600-second host timer gives 4459.683035284 for planning only:
old receipts precede ZIP construction, while the future timer includes it.
Separate publisher and job time, scheduling and cleanup overshoot are not thereby
bounded. Setup host-accumulator and child intervals differ, so no exact combined
setup total is invented. The successful zero-model smoke's resource observations
are listed separately and do not measure all other CI work. No per-run capacity,
acquisition, memory, output, wall or CPU threshold is raised by this proposal.


## Recovery 3 static activation history (now consumed)

Actual public-main base: `8c4d430ce1077aa9eefa8de130450319c64e545e`, tree
`e12cc9ac532b30057703261b741a1870b8f477ce`. The seven-file static activation delta
uses local disabled preparation `fbeccd8913502307aec11004f6c00a387884fd58`, tree
`6c527bc35ec7491711cf7c77c44fb11bb8626bcd`, as its separate comparison baseline;
that local snapshot was not public main. The consumed single native source is
`research/qwen12-native-admission-recovery3`, with create-only evidence branch
`evidence/qwen12-native-admission-recovery3` and directory
`research/qwen12-native-admission-v1/evidence/recovery3`. This exact source name
joins the independent cleanup retention list without changing its existing
entries. All immutable old source/run/evidence identities and surviving refs
remain preserved. The deleted original source ref stays consumed and must not
be recreated.

The fixed release and campaign contract bind all three prior immutable runs,
their exact source heads, evidence commits, archive hashes, first attempts and
0 started / 0 completed / 36 not-run records. A new identity does not reset the
same-twelve cumulative 36-recognition limit. There is at most one proposed new
claim, no automatic retry, no unbounded continuation and no inherited success.
Recovery 1 cleanup remains unconfirmed; the later successful cleanup and smoke
do not repair that historical unknown. All three current evidence archives
remain unchanged.

The [recovery 3 resource proposal](recovery3-resource-proposal.json) remains the
immutable disabled-stage budget snapshot, SHA-256
`5010e980e75b590fa62331a5bf1fd5ad1e65d916c093a9789eaf2ce4106003d9`.
Its false and pending fields describe that historical proposal, not a later
execution approval. It separately records three prior native observations and one
unchanged per-run reservation
for each measurement scope. It does not add ASR response bodies, Whisper tunnel
traffic, image-interface observations and audio response bodies into an alleged
wire-byte total. Recovery 1 audio and its never-entered Whisper stage retain
their source-control-flow inference qualifications. Paused-cgroup CPU excludes
host/image/final packaging/publisher and post-pause cleanup; sampled thresholds
have no unconditional finite overshoot bound. Old host receipt endpoints and a
future host timer covering final packaging differ; their sum is planning only.
Setup child and host intervals are not silently added. Separate smoke and
synthetic CI resource use is not included as an invented zero or all-CI total.
The historical recovery 2 proposal remains byte-for-byte unchanged.

The disabled preparation changed only native identity constants, the exact
three-predecessor roster, synthetic contract fixtures, source metadata and its
resource proposal from the public base. This static delta changes only both
workflow gates/status comments, release approvals and freeze binding, the
disabled-state test fixture, this README and append-only source-retention
maintenance/current native workflow hash. The committed-source state and all
strict gate/approval negative tests remain required.
Scientific/input/worker/runtime, host cleanup, full stress fixture,
ZIP/chunk codec and publisher algorithms and budgets are unchanged. The current
12MiB limit must retain complete evidence; it permits no dropped trace, source,
completion or failure records. No admission, independent evaluation or training
result is implied by successful preparation or synthetic transport.

At that reviewed activation, source, budget, independent review, selected
bounded-transport readback and historical-uniqueness approvals were literal JSON
true, together with `enabled`; bounded publication permission stayed true. The three prior
zero-use records, cumulative limit and exact runtime prerequisite roster remain
unchanged. Static approval does not mark any future runtime prerequisite passed.
The bounded-transport approval uses the completed live 12MiB proof at its exact
public source with unchanged codec/publisher algorithms. Recovery 3
subsequently executed once and stopped as recorded below.

The source-review PR, protected merge, fresh-main checks and immediate history
checks preceded that single execution. Its existing source/evidence refs and all
four immutable run/source/evidence identities must remain preserved. The current
source retains that consumed identity in its immutable history; it cannot
authorize its re-creation or another attempt. The separate fixed remaining35
contract below replaces the zero-only authorization semantics and excludes the
consumed slot. Its static activation remains subject to final execution checks.

## Terminal result: recovery 3, run 38173984595

[Run 38173984595, attempt 1, job 114572506731](https://github.com/jiying2007/kws-pipeline/actions/runs/38173984595/job/114572506731)
executed source `80009f412987aeaaaf401894677299724ed0490d`. Both runtime installs,
three prepare-only loads, whole-policy freeze and actual complete-closure stress
preflight passed; the latter measured 10,184,728 compressed bytes. The first fresh
Qwen worker then recorded `clip-000001` started and failed in configuration
observation. The retained bounded stack identifies `generation_snapshot` and
`clone`; the original exception message and failed live object are unavailable.
There are 1 started, 0 completed, 0 transcripts, 1 error and 35 not-run slots.
A start conservatively consumes a slot; it does not establish completed acoustic
model forwarding. This slot must never be rerun or reset to not-run.

All 55 controlled containers have verified cleanup and removal. Recorded CPU is
630.456231 seconds through acknowledged cgroup pauses; capture-receipt wall is
912.944030658 seconds, including successful predecode stress packaging but
excluding the later final evidence ZIP and publisher. Distinct acquisition
scopes remain separate. The recovery 1 historical cleanup unknown is unchanged.

Immutable [evidence commit 8c25805e226ca23d55bd698b52ba5b45acd9195a](https://github.com/jiying2007/kws-pipeline/tree/8c25805e226ca23d55bd698b52ba5b45acd9195a/research/qwen12-native-admission-v1/evidence/recovery3)
retains all 17 chunks and two metadata files. Full readback reconstructed the
4,210,336-byte ZIP, SHA-256
`9d371985530e8d5e32c915b8853f6e945ded1bfdbb6dad0b53fcbba4772e5bf5`,
with Git blob, chunk, member SHA/CRC and actual source/run/attempt/job checks.
The [public terminal metadata](evidence/run-38173984595/terminal-status.json)
binds selected original machine records without rewriting any old evidence.
Custody success establishes no transcript, admission or training result.

### Source-only active GenerationConfig observation repair

A separate zero-model replay used exact retained Transformers 4.57.6 method ASTs,
actual prepared JSON and synthetic Tensor fixtures. `_prepare_special_tokens`
adds `_bos_token_tensor`, `_eos_token_tensor`, `_pad_token_tensor` and
`_decoder_start_token_tensor` before the logits hook. `to_dict` retains these
runtime values, which the old strict JSON snapshot cannot serialize. This is a
source-backed reproduction consistent with the original bounded stack, not a
recovered original exception message or observation of its failed live object.

The active-only projector validates those exact four runtime fields against the
frozen token IDs. Tensor observations require exact native Tensor type, CPU,
int64, expected shape and values; BOS and decoder-start are absent for the pinned
request. The effective PAD fallback must equal the first EOS token. Their actual
JSON-safe observations are retained. No other field is filtered or stringified:
all remaining public and private configuration fields still undergo strict JSON
validation and complete frozen-snapshot equality. Only the locked native
max_length derivation from the independently observed prompt length is permitted.
The prepared generation JSON semantics and identity, live config, generation
method, request and native termination semantics remain unchanged. Changed
adapter source is bound by new source/native-config locks and source freeze;
those enclosing identities are not claimed equal to the historical run. Missing fields, drift, non-JSON objects,
NaN/infinity, fake tensors and incorrect token tensor metadata fail closed.

## Fixed remaining35 static activation history (now consumed)

The selected plan keeps every original 12×3 slot. The historical Qwen
`clip-000001` error is permanently consumed; exactly the other 35 slots are
eligible for a later single claim: Qwen clips 000002–000012 (11), SenseVoice
clips 000001–000012 (12), and Whisper clips 000001–000012 (12). The two other
clip-000001 outputs remain original planned machine observations and cannot
complete that source's three-recognizer chain. This plan does not select or
remove clips based on future recognition results. Any failure stops subsequent
work without retry or a replacement call; 35 is a maximum, not promised success.

The consumed source is `research/qwen12-native-admission-remaining35-v1`, with
create-only evidence branch `evidence/qwen12-native-admission-remaining35-v1`
and directory `research/qwen12-native-admission-v1/evidence/remaining35-v1`.
The exact source name joins the separate cleanup retention roster. All four
consumed source/run/evidence identities and surviving refs remain preserved;
the deleted original source ref must never be restored. Both native workflow
mirrors require the exact canonical public repository, fresh fixed-branch push,
first run attempt and successful contracts. The historical activation set six
native approval/enable bits true. This repair restores all six to false and the
literal workflow stop; prior bounded publication permission remains true.

The new fixed consumed-slot contract replaces the old zero-recognition-only
release field. It binds all four archive/source/run/attempt identities, their
original started/census references and the exact consumed family/WAV/PCM key.
Original audio IDs, source IDs, sample count, blind job order and all twelve
public input identities stay fixed. Both worker selection and receipt writing
reject the consumed Qwen slot or an unknown/substituted source. The count of
actual new keys must equal the selected family roster, rather than an assumed
12 per family. A historical reference does not create a started receipt in the
new run or reuse any old recognizer output under its new policy.

The cumulative terminal denominator remains 36. Its old Qwen slot is a distinct
historical error linked to the original immutable archive/member hash; new
slots report only their own attempts, completion or not-run state. Maximum new
starts is 35 and maximum historical-plus-new starts is 36. A fully completed
remaining plan means 35 new completions plus one historical error, never
all36-success. Intermediate failures preserve all consumed starts and all
unentered rows. The source contains no configurable resume cursor, generic
retry scheduler, fallback clip or automatic next execution.

All three prepare-only loads and the full frozen policy still precede any new
audio forward. The full unchanged stress fixture must fit 12MiB compressed,
32MiB expanded, 4096 members and 8MiB per member; complete failure/native records
are never omitted. Models, scientific inference parameters, token budget,
thread/CPU/time/memory controls, stop/completion validation, cleanup mechanics,
ZIP/chunk codec and publisher network algorithm are unchanged. Only their fixed
native destination identities change. The actual old 12MiB readback remains
bound to its historical head; new-identity transport proofs are local fixtures.

At most 11 sources could yield complete three-ASR chains, and actual admitted
counts may be smaller or zero. Each of the four frozen original groups still
needs at least one genuinely admitted matching source; completed captures alone
provide no coverage credit. Changed categories cannot replace original-group
support. Every preset/group stratum, including the permanently incomplete cell,
remains reported. Missing any original-group support stops training. Actual
source/rights/native evidence verification and separately reviewed train-root
registration remain required; this proposal makes no source train-admissible.

The [remaining35 resource proposal](remaining35-resource-proposal.json) records
four historical observations and one unchanged per-run reservation separately
for each measurement scope. Historical budgets and receipts are immutable. Mixed
network counters are not summed into wire bytes, sampled CPU/workspace thresholds
are not atomic quotas, and recovery 1's unknown cleanup and inferred acquisition
entries remain qualified. Separate smoke and synthetic CI observations remain
separate. No actual future run ID, head, elapsed resource use or result is invented.
This 40,104-byte proposal remains the immutable disabled-stage budget snapshot,
SHA-256 `14395223cd05a2dc1c2b5627e6a27d16500a0131a86530a272cb8e3e3915997d`.
Its pending/disabled fields describe that earlier proposal; current static
approvals are recorded only in the release and this source state.

The six-file static activation delta changes only the two workflow conditions
and comments, release approvals/freeze binding, regenerated source freeze, this
README and append-only source-retention metadata. Runtime and test bytes, fixed
four-predecessor contract, original 36-slot denominator, historical error, 35-new
maximum, resource proposal and coverage contract are unchanged. The actual live
12MiB readback proves the same publisher/codec at its historical identity; the
new remaining35 identity has only a local near-limit roundtrip proof. Retained
old prepare bytes plus complete current-source reserves fit the unchanged full
stress locally; every real run must still pass its actual closure preflight.

Those static approvals preceded the single remaining35 execution below. They
do not authorize reusing its source or evidence identity. Any later continuation
would need a separately reviewed fixed claim and five-predecessor ledger, exact
source PR/protected merge/fresh-main CI and immediate historical-consumption and
new-ref checks. That monitoring-repair checkpoint preserved the historical four-predecessor
contract as consumed-release context. The subsequent fixed five-predecessor
continuation is described below; its execution remains subject to final checks.


## Terminal result: remaining35, run 38180060677

[Run 38180060677, attempt 1, job 114590686389](https://github.com/jiying2007/kws-pipeline/actions/runs/38180060677/job/114590686389)
executed source `7dcb09208798298ec5a7e2d44b6c5611524fe0ba`. It stopped during
ASR runtime setup supervision, in the host wait phase with `OSError`. The
`install_runtime_wheels` command has a started record but no retained return code;
this does not establish a pip failure. No prepare, projector or recognizer ran.
Current starts/completions are both zero. The original 36-slot census remains
one immutable historical Qwen error plus 35 not-run slots; the historical reference
is not another new start and must never be double-counted or reset.

The one setup container has verified pause, final CPU close and removal. Its
controlled-cgroup CPU through pause was 18.847088 seconds. Host-receipt wall was
32.97841454 seconds, excluding the later final ZIP and publisher. The final ASR
response-body counter was not retained, so that exact total remains unknown.
119 retained wheel identities do not supply an exact download total or wire total.

Immutable [evidence commit 51f5d07c0ebf2ebab5bf9580347ab81cd0ea7a51](https://github.com/jiying2007/kws-pipeline/tree/51f5d07c0ebf2ebab5bf9580347ab81cd0ea7a51/research/qwen12-native-admission-v1/evidence/remaining35-v1)
retains a 42,908-byte ZIP, SHA-256
`2ecd301f1da3d1bcd5ffa5292fd1b60bad51b6aabfaad3f544c42214cf248c39`.
The original exception lacks errno and traceback; its precise cause remains
unknown. A separate zero-model replay of the original host source reproduced
real concurrent unlink, rename and symlink races between its metadata predicates
and later stat calls. That defect is compatible with the observed error, not a
recovered original traceback or proof of its unique cause.

### Disabled mutable-workspace monitoring repair

Resource sampling now opens directories without following symlinks and checks
same-open device/inode identity. Each entry has one nofollow metadata observation;
regular files and symlink inode sizes are counted without reading link targets.
Fixed public child scopes resolve from an identity-verified workspace descriptor.
The owned workspace identity is stable; optional child identities are not pinned
across legitimate output-to-frozen staging and recreation. An initially absent
optional scope is explicit zero at that observation, but a previously observed
scope that disappears fails closed.

Every entry/directory visit checks the same host deadline. Each byte addition
checks the same remaining scope threshold, and already-observed excess stops
immediately. Only genuine ENOENT during a mutable sample can produce an incomplete
byte lower bound. Such a sample may be followed by at most one fresh sample within
the same deadline; another incomplete sample stops. Two samples are never added
together. Non-ENOENT errors, type/identity replacement or an exhausted deadline
stop without another scan. Sustained concurrent installation may therefore still
stop safely. File-descriptor cleanup closes once and cannot replace the primary
scan error; separate bounded close diagnostics remain recorded.

The existing host receipt records fixed-scope cumulative counters, the latest
at-most-two observations, completeness/absence, selected sample and failure state.
Sampling is not atomic, a peak-size measurement or a kernel quota. Blocking
filesystem calls and scheduling do not gain a new hard per-call timeout claim.
Workspace/output thresholds, CPU accounting and all global time budgets remain
unchanged. Source, input, hash, lock and evidence integrity reads retain their
strict behavior and receive none of this mutable-sampling tolerance.

Exception projections add only a fixed errno allowlist and at most eight frames
from the actual executing frozen host code objects, with all external frames
redacted. They inspect neither error messages, paths, source lines, locals,
arguments nor environment. Primary, descriptor-close and Docker-cleanup failures
remain separate. Diagnostic failure cannot manufacture success or replace the
primary exception. Models, audio, token limits, frozen-config rules, stop/native
completion, the historical consumed slot and training coverage are unchanged.
This repair remains unchanged in the reviewed static continuation below and
establishes no additional recognition, admission or training result.


## Fixed remaining35-r1-v1 static activation history (now consumed)

This proposal uses fixed source `research/qwen12-native-admission-remaining35-r1-v1`,
create-only evidence branch `evidence/qwen12-native-admission-remaining35-r1-v1`
and directory `research/qwen12-native-admission-v1/evidence/remaining35-r1-v1`.
The new source name is separately retained by exact name. No remote ref or new
run is created by source preparation. All five consumed source/run/evidence
identities and existing refs remain preserved; the deleted original source ref
must never be restored. The source baseline is public main
`7dcb09208798298ec5a7e2d44b6c5611524fe0ba`; the reviewed disabled monitoring
repair is included without further changes to its runtime implementation.

The fixed `qwen12-fixed-remaining35-r1-v1` contract binds all five immutable
archives, original started-member rosters and terminal hashes. The fifth archive
has zero actual started members and current starts/completions zero, while its
cumulative count is one. Its one inherited error must point to the exact same
recovery 3 started member and hash. It contributes no second consumed slot.
Old four-predecessor releases, old source identities, changed fifth-run counts,
missing/duplicated historical references and altered archive bindings fail closed.
The original four rows and the unique historical-slot identity are preserved.

The planned new roster remains exactly Qwen clips 2 through 12, all 12 SenseVoice
slots and all 12 Whisper slots: 11/12/12, at most 35 new starts. Both selection and
started-receipt emission reject the consumed Qwen slot. The denominator remains
36 with one permanently historical error; no old receipt is copied as a new
attempt. The unchanged first-failure stop, frozen input/policy/native-completion
checks and original four-group training coverage still apply. At most 11 full
three-ASR groups can be produced; actual admission remains unestablished.

The [five-run resource proposal](remaining35-r1-resource-proposal.json) remains
the immutable disabled-stage planning snapshot, SHA-256
`dbde171c804b38ebc7bd5cf4f91423dceceb051ee27ac2211e2a94d15b6d96c0`.
Its disabled/pending fields describe that earlier proposal, not the current
static release or an actual execution result. The earlier
[remaining35 proposal](remaining35-resource-proposal.json) remains byte-identical.
Five pause-bounded controlled-cgroup CPU observations total 1684.135749 seconds;
host-receipt intervals total 2564.994306763 seconds. The latter exclude final ZIP
work whereas the future 3600-second host timer includes it, so their addition is
planning arithmetic with different endpoints, not a whole-workflow ceiling.
Image observations total 230572653 bytes. Qualified measured-plus-source-flow
figures are 3881040 audio bytes and 2322174334 Whisper tunnel bytes. These scopes
are distinct and never summed into network wire bytes.

The first four ASR response-body counters total 13363865788 bytes. The fifth
final counter remains unknown. Source-backed control flow bounds that named
Downloader counter to [523723580, 3340966448] bytes, and its five-run aggregate
to [13887589368, 16704832236]; these are derived intervals, not actual measured
values. The fifth upper bound is the sum of 137 pinned expected input sizes plus
one terminal excess byte, since both input loops share one fail-closed Downloader.
It excludes response/error/transport overhead and third-party process network
traffic. Local-wheel/offline-configured installation is not kernel network
isolation; total ASR network and all-wire traffic remain unknown. Recovery 1's
cleanup uncertainty and all other historical unknowns remain unchanged.

Future per-run reservations, sampled thresholds and wall limits stay unchanged:
4GiB named ASR acquisition counter, 1.5GiB Whisper tunnel, 128MiB image observation,
1MiB audio acquisition, 6600 sampled cgroup CPU seconds, 3600-second capture-host
timer and 10GiB sampled workspace. The full evidence cap remains 12MiB compressed,
32MiB expanded, 4096 members and 8MiB per member, in at most 48 chunks. These are
scope-specific controls, not an atomic resource or wire quota. The historical
cleanup smoke proves only its unchanged cleanup component at its executed head;
no new-head live smoke is claimed. The actual 12MiB live readback remains bound
to its old source identity. This new destination has local synthetic roundtrip
proofs only; the publisher and codec algorithms are unchanged.

The six-file static activation changes only both workflow conditions/comments,
six strict native dispatch bits, source-freeze identities and binding, this
README and append-only retention metadata. Runtime, tests, fixed five-predecessor
contract, resource proposal, source/input identities and all limits remain
byte-identical to the independently reviewed disabled checkpoint. Actual
bounded-transport approval uses the prior live12MiB proof with unchanged codec
and publisher algorithms; the new identity has only a local near-limit proof.

Source, budget, independent review, bounded-transport readback and historical
uniqueness have static approval, with `enabled=true`; publication approval remains
true. Every runtime prerequisite remains required for the actual run, never
pre-marked passed. Before execution the exact source must pass source PR,
protected merge and fresh-main CI, followed by immediate checks of all five
consumed histories and unused new source/evidence refs. Only root may then
create one fixed fresh source ref at that approved main. No execution is performed
by this static preparation. There is no automatic next identity or replay mechanism.


## Terminal result: remaining35-r1, run 38185393757

[Run 38185393757, attempt 1, job 114606697135](https://github.com/jiying2007/kws-pipeline/actions/runs/38185393757/job/114606697135)
executed source `0f0f736c6faaecb0c62dffd6ef5e3707f1f4a894`, tree
`df77173a0a8191326a230b60f866cdd84293d3a8`. All 35 permitted new captures have
actual started/completed receipts and nonempty native transcripts: Qwen11,
SenseVoice12 and Whisper12. The original denominator remains36:35 native-complete
plus the unique historical Qwen error/null, no current errors or not-run slots.
The cumulative36-start allowance is exhausted. Workflow success, native completion
and label admission are separate outcomes.

Immutable [evidence commit38dd722574abef198ee97b424cb8864599d0c713](https://github.com/jiying2007/kws-pipeline/tree/38dd722574abef198ee97b424cb8864599d0c713/research/qwen12-native-admission-v1/evidence/remaining35-r1-v1)
retains18 chunks plus2 transport metadata files. They reconstruct4,692,291 ZIP
bytes, SHA-256 `f07f44e08c84eb9d5d70699a82eb848b334597df73fa1210a327aeae39d0983d`.
All1,005 payload members plus the manifest, original source/run/job bindings,
Git/blob/chunk/member hashes, CRC and native completion/source/config closure
were independently verified. No original source or evidence bytes were rewritten.
The [public derived metadata](results/run-38185393757.json) retains the full12-row
machine transcript matrix and exact original member references without private
human-review bodies or model weights.

Of11 complete three-ASR sources,6 are unanimous under the unchanged fixed
punctuation/whitespace removal,3 have only two-way agreement and2 all differ.
The incomplete historical-error source has only SenseVoice/Whisper observations
and cannot acquire a third vote. The6 unanimous candidates are K2 clips3/12,
repeat-nihao clips2/6/7 and suffix-kw2 clip8. They are machine candidates only:
source/rights/history resolution and formal train-only admission were not issued.
Acoustic verbatim completeness remains unknown, and no human-gold claim is made.

All three original K1 sources fail strict three-way agreement:

| Preset | Qwen06 actual | SenseVoice actual | Whisper actual |
|---|---|---|---|
| Dylan | 您好，小吴。 | 您好小啊 | 你好小巫啊 |
| Uncle_Fu | 你好，小沃。 | 你好小沃 | 你好小我 |
| Vivian | 你好，小屋。 | 你好小我 | 你好小屋。 |

These contain out-of-vocabulary characters and cannot be repaired by mapping
沃/屋/我 to窝, two-vote selection, replacing unknown human observations, or blank
negative targets. The original four-group candidate upper bounds are0/2/3/1;
K1 cannot reach its frozen minimum1 even before any additional admission rejection.
The result is `INSUFFICIENT_ADMITTED_COVERAGE`: training, export and model/C readback
were not run. The frozen coverage/recipe and all original members remain unchanged.
A separately approved future data experiment cannot be inserted into this old
frozen group roster or presented as completing this failed coverage condition.

All57 controlled containers have acknowledged final CPU accounting and verified
removal, with no cleanup/receipt errors or sampled CPU-threshold violation. The
controlled-cgroup CPU through pause was970.499524 seconds; host-receipt elapsed
was989.8504755309999 seconds, excluding the subsequent final ZIP and publisher.
Separate byte scopes were3,340,966,447 ASR response-body bytes,774,054,311 Whisper
bidirectional userspace tunnel-recv bytes,45,687,055 default-interface image
receive-delta bytes and776,208 audio response-body bytes. They are not added into
an alleged wire total. The8 filesystem scopes retained complete latest samples
with no latest vanished entries; the actual errno/frame error path was not
triggered. Sampling is not atomic, a peak measurement or a hard quota, and this
successful run does not retroactively resolve recovery1 cleanup uncertainty or
the fifth run's missing ASR final counter. Whole-CI resources remain unmeasured.
