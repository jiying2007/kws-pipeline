# Bounded lifecycle v2: source-only candidate

This is a new, self-contained Linux/stdlib controller candidate. It does not
modify or import any historical controller, archive, authorization, model or
result. Existing failed budgets remain failed. It has not been connected to a
TTS, ASR, training, Library, or production execution adapter.

## Actual entry points

`Controller.create` freezes a unique item roster and storage policy.
`Controller.execute(callback, per_item_bounds=...)` is the new serial execution
entry; it durably claims each item **before** callback invocation.
`Controller.deliver(callback, peak_bounds=...)` is a separate delivery entry.
The tests invoke both, not isolated arithmetic helpers.

Run the integrated CLI fixture from a writable parent directory with at least
2 GiB plus the bounded fixture headroom free:

```sh
python3 -B controller.py fixture --root /tmp/kws-lifecycle-new-fixture
python3 -B controller.py status --root /tmp/kws-lifecycle-new-fixture
python3 -B -m unittest discover -s . -p 'test_controller.py' -v
```

The fixture must receive a **new** directory. It performs 16 artificial callback
calls and writes tiny files in all five categories. The `upload` file is an
artificial local staging copy: **no remote upload occurs**. It verifies local
copy byte equality. Tests also invoke a subprocess that abruptly exits after
consumption, then verify that reopening cannot execute the operation again.

## Budget and count semantics

- Every action supplies bounds for `output`, `archive`, `upload`, `readback`, and
  `temp`; missing bounds, booleans, negative or non-integer values are rejected.
- Admission uses logical file lengths (not allocated blocks), all declared future
  category bounds, and a 128 KiB control allowance. Each guarded write rechecks the total
  budget and physical free space. Failed/partial artifacts remain charged.
- The disk floor cannot be less than 2 GiB. External tool download temporaries,
  local upload staging, ZIP overhead, and remote readback copies must be included
  in declared bounds and brought under the adapter's guarded write contract.
- Control state is limited to 48 KiB; at most 256 roster items and 64 delivery
  attempts are accepted. Old+new state fit the conservative control allowance.
- `planned` is the frozen roster length. `consumed_or_unknown` counts items
  durably claimed before callback; `completed` counts callbacks returned without
  exception; `unknown` includes abrupt death and failed callbacks. These are
  **callback-operation counts**, not inferred model-forward counts. An adapter
  doing nested reference preparation/embedding/generation must expose those
  operations in its own explicit bounded roster/counters.
- Budget admission is not a lifetime reservation against other processes. An
  execution may fit while a later requested delivery does not; that delivery is
  rejected before callback and the completed execution remains intact.

## State and retry semantics

Execution and delivery states are independent. A failed model callback remains
consumed/unknown, and neither failed nor completed execution can be retried by
this controller. Failure evidence can be delivered without making execution
successful. A completed execution remains completed if delivery fails.

Delivery exceptions are conservatively `unknown`, because a remote effect could
have happened before the error. Automatic retries are blocked. The operator must
actually reconcile remote state first. `reconcile_delivery` accepts a bounded
reference to that evidence and verified remote absence; it does not authenticate
or obtain that evidence and is not permission to send. Retry creates a fresh
attempt directory, retaining earlier artifacts. If remote completion is found,
manual reconciliation/review is needed; do not pretend absence or resend.

`flock`, exclusive-create artifact writes, fsync, and atomic state replacement
prevent simultaneous local callback execution and partially published state. An
unfinished `state.pending` file is retained and fails closed on the next write;
there is no automatic recovery/deletion. Process death after claim is reported
as unknown. This controller does not launch, supervise or kill model processes.

## What is not proved

This is **cooperative I/O admission**, not a filesystem quota, cgroup memory
limit, adversarial sandbox, or absolute disk-floor guarantee against concurrent
external writers. Callbacks must only write artifacts through `Writer`, must not
spawn unmanaged writers or retain the writer after returning, and must correctly
bound tool-internal copies. Inputs and immutable runtime/model storage are outside
this action's byte cap but still reduce observed disk free space. `upload` accounts
for local transfer staging, not remote storage quota or network-byte billing.

The local state directory must be on a trusted, persistent filesystem. This
module is not a remote durable journal, secure multi-user service, evidence
signature, permission system, or replacement for source/input hashes, complete
process-group cleanup, numerical gates, and model-specific authorization. The
`verified` receipt flag is a caller assertion; real adapters must establish the
remote identity/readback evidence before setting it. `Writer.write_stream` reads at most 64 KiB per request and requires an exact
known byte count; oversized/short sources fail with their partial file retained.
It avoids whole-archive loading for conforming file readers. The convenience
`write(bytes)` API and allocations inside callbacks are not memory-budgeted. The disk-free injection argument is for tests; production
CLI always reads `shutil.disk_usage`.

## Safe migration for a future authorized run

1. Keep all old controllers, state, ZIPs and historical corrections unchanged.
2. Bind a new run ID, approved operation roster, source/model/input identities,
   and complete execution+delivery upper bounds. Keep model and external-send
   authorizations separate from local controller state.
3. Implement one real adapter through this new entry point. Preserve the existing
   before-call consumption marker, nested model counters, process supervision,
   failure evidence and cleanup checks. Do not treat a returned callback alone as
   acoustic/model success.
4. Add adapter-specific fault injection: partial write, oversized archive, download
   temp peak, unexpected child, process death, state failure, uncertain remote
   upload, and successful remote reconciliation. Verify that rejected budgets
   suppress the actual write/tool call and never cause a second model call.
5. Establish persistent local storage and a small verified remote journal if
   cross-environment recovery is required. Local fsync cannot replace remote
   durability. Do not skip the old protocol's Library proof requirements in place.
6. For the new approved protocol only, advance execution on verified durable
   local terminal/cleanup evidence; package and deliver complete user attachments
   separately. Retry only verified-safe delivery, never consumed inference.
7. Only after integration review and authorization execute a bounded real canary.
   This candidate's artificial tests do not establish production readiness.

## Separate small follow-ups

CI: extract research-change selection to a small job, keep all required aggregate
checks fail-closed, then compare critical-path timing. Move compiler-independent
navigation tests out of compiler matrices. Do not change CI in this candidate.

Retention: keep the existing ledger immutable as the historical base. A separate
migration may append small immutable receipts plus a current index, with validators
reconstructing and validating the full projection. Do not weaken historical byte
checks or rewrite old evidence. Neither change promises removal of tool approvals.

### Filesystem accounting boundary

The total-byte cap is a logical retained/peak-byte admission policy, not an
allocated-block quota. The 128 KiB control allowance covers bounded JSON state
and an estimate of control overhead; it is **not proof** of filesystem-specific
inode, directory, journal, copy-on-write or allocation overhead. Physical disk
free space is re-read before guarded writes, but projected logical increments
plus that allowance cannot guarantee a hard floor on every filesystem. A real
adapter must measure/reserve the target filesystem's amplification or use an
OS/filesystem quota, with headroom above the policy floor.

The root and all ancestors must be trusted and not replaced concurrently.
Symlink checks catch ordinary accidental links; they are not race-free protection
against a hostile actor replacing parent directories or modifying state. This
candidate intentionally does not implement a new security sandbox.

## Historical-chain review provenance

The prior read-only audit inspected the historical controller's
`verify_reservation` / `prior_delivered` / `acknowledge_result` chain: each stage
required reserve/result Library readback, and result packaging included the
reserve archive. Its before-call consumed markers and process cleanup were
separate from those delivery prerequisites. The audit also observed a legacy
8-call summary reused by a 16-call task and archive/readback bytes excluded from
an original execution-output budget.

During this candidate's implementation, the old TTS/ASR controller source files
were **not available** in the current workspace or the recovered three evidence
packages. Their previous line-by-line audit is historical evidence, not a fresh
source revalidation here. This candidate is independently self-contained and
never imports those missing files. Before a real adapter migration, recover and
hash-verify the exact old source and repeat the adapter/cleanup comparison.

## Revision 2: explicit interrupted-execution sealing

Revision 1 is retained unchanged. Independent review identified two gaps fixed
here: a crashed execution could not deliver failure evidence, and a newly
created artifact-prefix directory needed its parent's directory entry fsynced.

After an interrupted controller, an operator may explicitly call
`seal_failed_execution(processes_stopped_verified=True, evidence=...)`, or:

```sh
python3 -B controller.py seal-failed --root /path/to/interrupted-state \
  --processes-stopped-verified --evidence 'reference to verified process-cleanup evidence'
```

This is permitted **only after independently establishing that the relevant
model and descendant processes have stopped**. An unlocked file is not that
proof. The method immediately rejects an active controller lock (nonblocking
exclusive flock), accepts only interrupted `started` execution, preserves every
consumed operation, converts started items to `unknown`, leaves completed/pending
items unchanged, and seals the run as `failed`. It never guesses model success,
resets a call slot, or enables execution retry. Failure evidence can then use the
separate delivery entry. The evidence argument is a caller assertion/reference,
not an authentication mechanism. An incomplete `state.pending` still requires
separate investigation; this method does not delete it or bypass fail-closed
state publication.

Artifact-prefix creation now fsyncs the category parent **before** file creation.
Tests verify ordering and inject failure at that sync point: no artifact is
written, the item remains consumed/unknown, and execution retry is forbidden.
The hard-exit regression runs the sealing CLI, delivers failure evidence, and
checks that subsequent model execution is still rejected.
