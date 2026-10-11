# Qwen12 one-shot controlled native admission capture

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
the exact identities, counts and evidence limits. Preserve the original source
and evidence refs; do not rerun or delete/recreate this consumed campaign.

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
ZIP into at most 32 raw chunks of at most 256KiB. Its small outer manifest binds
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

This reviewed recovery activation removes the native workflow literal-false stop
and sets the exact reviewed release enabled state. Dispatch still requires literal JSON true for
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
The unchanged sequential host path propagates any prepare/freeze/package failure
before its decode loop; fresh-worker verification remains mandatory. A static
release approval or synthetic fixture cannot replace actual runtime evidence.
The exact collector continues retaining those records; no new proof schema,
platform, model load or evidence omission is introduced.

## Activation handoff

The activation procedure below records the original pre-run requirements.
The source was reviewed, passed exact-head CI and was merged before the one-shot
campaign ran and stopped as recorded above. Its source and evidence refs now
exist and must remain unchanged; this historical procedure does not authorize a
second run. Source review PRs run read-only fixtures and cannot run either
contents-write publication job.

The zero-model repository-chunk preflight has passed actual connector readback:
[run 38152411613, attempt 1, job 114508486242](https://github.com/jiying2007/kws-pipeline/actions/runs/38152411613/job/114508486242)
executed source `d33107c825511b5be994e6e922f31d435f89d808` and created
[evidence commit 46e605153482d7babd7584fdc960c15022be8c12](https://github.com/jiying2007/kws-pipeline/tree/46e605153482d7babd7584fdc960c15022be8c12/research/qwen12-native-admission-v1/evidence/synthetic-v1).
All 34 files (32 chunks and two metadata files) were read at that immutable
commit; Git blob identities, chunk hashes, ZIP members/CRCs, fixed synthetic
seed, independently fetched run/job metadata and checkout identity passed.
The reconstructed ZIP is 8,326,571 bytes with SHA-256
`0e2b26a972a06dee180ac0a4f6cad13166164a64e8c4918c181b75d086e4f466`.
This proves the selected bounded repository transport. The previous large-log
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

## Recovery 1 reviewed activation source

The original execution and evidence refs remain immutable. A separately reviewed
single continuation may use `research/qwen12-native-admission-recovery1`, with
create-only evidence ref `evidence/qwen12-native-admission-recovery1` and directory
`research/qwen12-native-admission-v1/evidence/recovery1`. The native workflow literal stop is removed and all static release approvals
are enabled in this reviewed source. No recovery execution has occurred. Before activation, recheck the predecessor's zero recognition
consumption, absence of the new refs/run, exact source CI, budget and independent
review. `execution-release.json` binds the original run, source, evidence commit,
archive hash and 0/0/36 accounting. Across both executions, the same twelve clips
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
