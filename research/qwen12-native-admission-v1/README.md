# Qwen12 prospective native admission (disabled)

This new scope consumes exactly twelve pre-existing, public Qwen WAVs from the
pinned `kws-data` commit. It does not regenerate audio, reuse historical Qwen16
outputs as Qwen12 results, or change any historical freeze, gate or evidence.
Only a hash-ordered anonymous twelve-row job and its audio enter recognizers.
Setup and inference see only explicitly frozen public code, locked model/runtime
assets and safe run/head provenance. No human labels, target text, peer output,
GitHub token or host environment is mounted/passed into a worker.

## New executable path

`hosted_run.py` uses the existing official Python image/host-capacity/kernel
scope checks. The fresh `setup_adapter.py` is the reviewed source-screen setup
binding copied to this new scope with only task workspace/initial-free limits
changed to 10/12GiB. It retains 124 locked Qwen/Sense package inputs (119 wheels,
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
  each clip120s outer host threshold; whole host3600s plus cleanup/log delivery.
  Scheduling and cleanup can overshoot sampled wall thresholds; this is not a
  real-time kernel timer guarantee.
- Cgroup cumulative CPU sampled threshold: setup3600 + Qwen600 + Sense600 +
  Whisper1800 =6600CPU seconds. All controlled cgroups are charged from creation.
  A new minimal gate keeps each cgroup alive before and after its child so the
  host can meter it, then acknowledge pause and capture usage before removal.
  Receipts report actual poll gaps and missing/unconfirmed accounting. The scope
  excludes host, Docker image acquisition, log compression and cleanup after
  pause. No strict all-host, zero-overshoot cumulative CPU claim is made.
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
The capture host's 3600s timer and sampled 6600 CPU-second accounting end with
capture cleanup; they exclude the publisher's CPU and HTTPS traffic. Publication
has its own 180s supervised deadline and 4-minute step timeout, inside the existing
70-minute job ceiling. It never retries recognition or changes capture budgets.
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

The workflow literal false and execution-release enabled=false are independent
activation stops. Dispatch additionally requires literal JSON true for source,
budget, independent review, actual complete selected-transport readback, historical campaign
uniqueness, and bounded public transcript publication approvals. Missing, false,
integer or string substitutes fail before Docker, network or model imports.
Actual workflow run_number is preserved, not invented or required to be1.

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

This committed tree is source-only. Create its source PR on
`fix/qwen12-native-training-capture-20261011`; do not create the campaign branch
while source CI, independent review, selected-transport readback or history review
is pending. The source PR runs only read-only stdlib fixtures. It never runs a
contents-write publication job. After source CI and independent review, create the separate
`research/qwen12-native-transport-preflight-v1` branch once at the exact reviewed
source commit. Only its canonical public-repository push with event.created=true
and run_attempt=1 runs the zero-model maximum-size synthetic transport job, which
also needs the contracts job. The fixture creates only
`evidence/qwen12-native-transport-synthetic-v1`, under the fixed directory
`research/qwen12-native-admission-v1/evidence/synthetic-v1`. Root must fetch its
near-8MiB package by immutable evidence commit through the actual connector and
verify the full archive before setting bounded_evidence_transport_readback_approved.
Large-log readback is not claimed as successful.

The real `research/qwen12-native-admission-v1` campaign stays independently
literal-false disabled. After all dispatch approvals and activated-state contracts
pass, its new-branch push may capture once and publish only to
`evidence/qwen12-native-admission-v1`, under
`research/qwen12-native-admission-v1/evidence/native-v1`. The real job also needs
contracts. Both evidence refs must be absent and are created once; neither is
updated or deleted/recreated. Root separately verifies source-branch/run history,
the unique campaign claim and the exact activated source head/freeze/retention.

The actual complete prepared closure and stress-package check is a same-run
pre-forward requirement, not an externally pre-passed dispatch approval. A
minimal-config fixture alone is not that proof. The host performs it after the
three prepare-only loads and stops before any forward if the package does not
fit; the existing 10GiB workspace checks remain in force.

After the dispatch reviews are closed, use a separately reviewed activation commit
(or a separate PR containing it). Remove only the workflow's literal false,
intentionally update source-only tests for the reviewed activated state, set all
reviewed release flags, update the exact new active-workflow bytes/hash/Git
blob in the source-retention manifest, regenerate execution-freeze.json, and
bind its new SHA in execution-release.json. Run all source/retention checks
again. Historical workflow/source/freeze identities remain unchanged. Source
hashes from the disabled commit do not describe the activated tree. Its final
commit is the new branch push head and the real Actions identity in all captures.

Before that push, root must inspect repository branch/run history and record a
single campaign claim: no earlier run may have consumed this twelve-by-three
campaign. Preserve event.created=true and run_attempt=1. Never delete/recreate
the campaign branch or rerun a failed campaign; a failure requires review of its
retained error/not_run evidence. Source-CI run_number is unrelated and is kept
as supplied by Actions. Do not create a training source root until actual native
results have independently passed the unchanged resolver and admission checks.
