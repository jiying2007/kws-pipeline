# Offline evidence safety toolkit (candidate v1)

This is a new, **saved-only research utility**. It does not modify the frozen
qualification v2 guard/helper, restart a failed attempt, load a model/native
library, process audio, or consume or issue any historical one-shot permission.
A successful synthetic test is not evidence that the historical method passed.

## What changed

`offline_safety.guard.save_then_compare` is the supported new review entry point:

1. Create an exclusive new directory and persist a nonqualifying STARTED marker.
2. Persist both raw ndarray snapshots, including NaN, Inf, wrong shapes, wrong
   dtypes, missing members and invalid counts. Flush/fsync, read back and verify
   exact dtype, shape and bytes; persist RAW-SAVED with `complete: true` and
   `qualified: false`. No numerical validity is implied by byte completeness.
3. Explicitly reject optimized Python (`-O`/`-OO`), disarmed calls and missing,
   incorrect or stale approval bindings using `if`/`raise`, never `assert`.
4. Load the caller-pinned public saved-only admission module. The dependency is
   not copied or edited. Execute its **real `require_admission`** against actual
   source, input, reference and dependency files; do not trust a supplied PASS.
5. Require the exact admitted arm, fixture, reference digest, source freeze,
   dependency manifest and ticket identity. Delegate the unchanged numerical
   rules to the public helper, using verified persisted bytes rather than mutable
   caller arrays. Verify the saved NPZ, identity and raw receipt hashes again
   before a final qualifying receipt. Re-audit once more before the final qualifying terminal.
6. Write an exclusive TERMINAL receipt. `complete` describes the raw snapshot;
   `qualified` is true only if admission and every numerical gate succeed.
   Failures retain raw evidence and a rejected terminal. The outer TERMINAL is
   authoritative; a nested saved/comparison receipt alone does not qualify a run.

The integrity approval binds the guard, public helper, fixed policy, ticket and
fixture identity hashes. Its scope is only SAVED_TRACE_REVIEW_ONLY. It is not a
cryptographic signature, independent authorization service, nor permission to
start any computation. The caller supplies an externally reviewed binding;
`approval_binding` only exposes its canonical schema. Default `armed=False`.
No CLI launcher is provided.

Raw NPZ members are written explicitly, so keys such as `file` and `allow_pickle`
remain evidence rather than binding serializer parameters. Exact member-name
readback also distinguishes `x` from `x.npy`. NUL-containing names cannot be
represented faithfully by ZIP and are refused with an incomplete rejected receipt.
Object-dtype arrays and non-ndarray objects cannot be safely persisted without
pickle and are explicitly refused, leaving STARTED/REJECTED evidence. Disk errors
can make evidence incomplete; no complete receipt is fabricated. An abrupt OS
crash can leave STARTED or RAW-SAVED without TERMINAL. Either is nonqualifying.
The final verdict is staged and file-fsynced before atomic, nonoverwriting
publication. A final directory-fsync error propagates: a published valid numerical
verdict can then exist, but durability is unconfirmed. Only successful function
return confirms the complete fsync sequence; pending files never qualify.

## Reusable pre-launch preflight (separate boundary)

`offline_safety.preflight.require_preflight` has no launcher. It rejects optimized
Python explicitly, defaults to disarmed, and recomputes an exact new-scope
approval binding over policy, verifier, named source files and named input files.
Missing approval, legacy once scope, source/input/policy drift and missing files
are denied before the caller may spawn. The synthetic process tests call this
preflight immediately before each real fixture subprocess; negative tests assert
zero launches, including `-O` and `-OO`.

The only accepted scope is NEW_OFFLINE_SYNTHETIC_PROCESS_ONLY_V1. It grants no
model execution. `preflight_binding` computes a proposed hash schema, not user
consent; an application must supply separately authorized approval and include
all relevant source/input files. It is a reusable future integration primitive,
not a retrofit of the historical guard. A future real protocol needs a separately
reviewed scope, execution authorization, isolation and resource enforcement.
Before/after drift protection is not an atomic filesystem snapshot. Keep bound
inputs immutable and call the preflight immediately before the separately owned
launch. Neither the preflight nor supervisor consumes an old once permission.

## Process failure evidence

`offline_safety.supervisor.capture` only monitors the stdout/stderr pipes of an
**already started, trusted synthetic process**. It has no launch function and
must not be treated as a replacement for a model-execution authorization guard.
The caller must separately authorize and isolate any process before starting it.

It retains exact raw byte prefixes, enforces a combined stdout+stderr byte cap,
uses one absolute monotonic deadline, bounds each select by its remaining time,
checks the deadline again after observing final pipe EOF and exit, kills/reaps its child on
failure, and persists terminal returncode/completeness/hash evidence. Timeout
and cap exhaustion never claim complete capture; explicit truncation identifies
cap exhaustion. Exit failure may have complete captured bytes but never success.
Even successful capture always has `qualified: false`; numerical admission is
separate. A crash/disk failure can leave only incomplete STARTED evidence.

This is not a sandbox, process-tree killer, CPU/RSS limiter, filesystem quota,
network restriction or protection against a malicious producer. Other child
output files are not covered by the pipe cap. Callers own process isolation,
pre-launch permission, descendant containment and cleanup on precondition error.
No claims about historical guard resource checks are changed.

## Dependency and test command

Python 3.11+ on POSIX and NumPy are required. Use the restored **public**
reference-admission-hardening/admission.py, SHA-256:

    f825a3e081723b8e153e6ec93a68b6a1d69acbc27c84937cefed9cf0d3dc900c

From a clean repository clone, the explicit --fetch-helper CI command downloads 8 immutable public archive
parts (59,417,250 bytes total, about 59.4 MB) into a temporary directory. Each
part and the concatenated ZIP are size/SHA-256 checked. It reads only the pinned
helper object from the ZIP and verifies that object again; it never extracts or
executes any other archive object, old guard, model or data file. The checked-in
PUBLIC-DEPENDENCY.json fixes repository commit, URLs, all part hashes and the
helper identity. No network fetch occurs without this explicit option:

    cd research/offline-evidence-safety-v1
    python -m venv .venv
    .venv/bin/python -m pip install -r requirements-test.txt
    .venv/bin/python run_tests.py --fetch-helper

For an offline run with a previously acquired copy of the same public helper:

    .venv/bin/python run_tests.py --helper /path/to/public/admission.py

Direct test invocation also works with PUBLIC_ADMISSION_PATH set to that file:

    PYTHONPATH=src PUBLIC_ADMISSION_PATH=/path/to/public/admission.py python -m unittest discover -s tests -v

The suite creates all inputs in temporary directories. Its complete synthetic
saved-audit layout runs the real 26-record/192-gate reference audit, including
real hashes and dependency files; there is no mocked audit. The `official-pcm`
fixture filenames match the historical schema but contain generated zeros and
perform no audio processing. Deterministic test hooks simulate disk-full failure and evidence-file substitution;
they do not mock the admission audit.
Pure Python subprocesses produce fixed strings, exit codes and delays; nested
`-O`/`-OO` tests invoke the actual saved-review entry point and must reject.

No old checkpoint, private corpus, private weight, authentication data or old
attempt output is used by these tests. Any locally supplied saved arrays remain
local; this tool never publishes data. Source publication requires its own scope.

## Limits

Hash bindings detect ordinary drift but do not protect against an adversary able
to rewrite the verifier, all inputs and approval. Inputs should be immutable for
the review duration: the before/after audits detect ordinary changes but are not
an atomic filesystem snapshot against concurrent malicious writers. Raw array
snapshots are deliberately saved before validation; the caller must provide
trusted, memory-sized ndarrays and enough evidence disk space. No global raw
snapshot size limit is claimed. Trust and pin the public Python dependency before
loading it. This candidate is independent of all historical frozen entry points.
