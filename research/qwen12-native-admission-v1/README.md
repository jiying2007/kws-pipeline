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
  fixtures do not establish real host compatibility; that needs preflight.
- Task workspace10GiB is a sampled logical-tree stop threshold, not a filesystem
  quota. Docker image/layers and tmpfs are outside it. Output32MiB is a sampled
  aggregate stop threshold; container per-file fsize remains independently bound.
- Aggregate acquisition reservations: Qwen/Sense4GiB exact response bodies,
  Whisper1.5GiB bidirectional tunnel recv, image128MiB observed-interface reserve,
  audio1MiB exact response bodies. Sum5.626GiB <6GiB. These are deliberately
  different measured scopes; image cancellation does not prove daemon stop and
  this is not a hard total network-wire cap. No failed image attempt proceeds.

## Evidence and cost

No Actions artifact or cache action is present. New derived native transcripts,
completion/control/failure records, resource receipts and public source/tokenizer
closure are transported as bounded hash-bound log chunks after explicit approval.
The consumer must bind the real run/head/job API response, reject missing,
duplicate or mixed chunks, and revalidate exact archive members. No fake artifact
ID or fabricated run identifier is produced. User approved the bounded public
record scope; source/budget review and release remain separately disabled.

No weights, audio, private human reviews, raw environment, arbitrary error text,
or signed download URLs are collected. Complete trace/source dependencies must
fit the reviewed log package. Oversize evidence is a failure, never permission to
omit inconvenient failure, truncation, source or trace records.

## Review and verification

Run `python3 -B test_contract.py`, then the same with `-O` and `-OO`; tests perform
no network, model imports, acquisition or inference. Also run the sibling native
adapter/Whisper-binding/lock tests and log transport tests. `prepare_freeze.py`
runs only after all source is integrated in a separate checkout; it hashes the
exact source/role roster and does not enable execution.

The workflow literal false and execution-release enabled=false are independent
activation stops. A source CI/log-transport synthetic smoke and host-capability
preflight must not consume a real twelve-clip batch claim. Real execution needs
root-reviewed source, budgets, complete immutable freeze, an8MiBzip/32MiBexpanded maximum synthetic log
roundtrip read back through the actual GitHub log API, compatible real host,
and a fresh new branch push with run_attempt1. Actual workflow run_number is
preserved, not invented or required to be1.

## Activation handoff

This committed tree is source-only. Create its source PR on
`fix/qwen12-native-training-capture-20261011`; do not create the campaign branch
while source CI, maximum-log API readback, or independent review is pending.
The source PR runs only stdlib fixtures and public synthetic log transport.
The complete actual prepared runtime/source/tokenizer/config/inventory closure
plus the 12 × 224 full-prefix trace stress package must pass before activation;
a minimal-config fixture alone is not that proof. The host repeats this check
before any audio forward and fails closed if the 10GiB workspace cannot fit.

After these blockers are closed, use a separately reviewed activation commit
(or a separate PR containing it). Remove only the workflow's literal false,
set reviewed release flags, update the exact new active-workflow bytes/hash/Git
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
