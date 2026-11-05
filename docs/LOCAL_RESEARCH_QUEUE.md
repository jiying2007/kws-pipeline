# Local completion-driven research queue

`tools/research_queue.py` implements the local execution portion of
[`active-plan-20261105.json`](../configs/research/active-plan-20261105.json).
It uses the existing durable-claim, create-only receipt, `fsync`, process-group
timeout and `flock` patterns. It does not run model code during initialization,
import an ML runtime, call a paid service, publish results, create a remote
schedule or turn previous experiments into new queue results.

## Declare a bounded queue

The active plan caps each item at **999 minutes** and defines aggregate cost as
the sum of actual job minutes. It supplies no numeric aggregate ceiling. Each
queue therefore requires an explicit positive `global_budget_minutes`; there
is no invented default total. This budget covers all jobs in that frozen queue.
Different queue roots or earlier manual runs are not silently counted as part
of it. Keep one persistent root for the intended campaign and use the same
heavy-model lock path across all cooperating model entrypoints.

A request has this structure. Replace the example absolute paths and the
recipe digest with the actual declared files and SHA-256 before initialization:

```json
{
  "schema": "kws-local-research-queue-v1",
  "queue_id": "local-research-001",
  "global_budget_minutes": 5,
  "jobs": [{
    "job_id": "inspect-saved-inputs",
    "recipe": "/private/research/recipe.json",
    "recipe_sha256": "ACTUAL_64_HEX_SHA256_OF_RECIPE_BYTES",
    "argv": ["/absolute/path/to/python3.12", "-B", "/absolute/path/to/a-reviewed-entry.py", "--recipe", "{recipe}", "--output", "{output_dir}"],
    "cwd": "/private/research",
    "depends_on": [],
    "timeout_minutes": 2,
    "outputs": ["result.json"]
  }]
}
```

```sh
python3.12 -I -S -B tools/research_queue.py init \
  --request /private/research/request.json \
  --root /private/research/queue-001 \
  --heavy-lock /private/research/heavy-model.lock
python3.12 -I -S -B tools/research_queue.py run --root /private/research/queue-001
python3.12 -I -S -B tools/research_queue.py status --root /private/research/queue-001
```

The root must be new. The declaration, active plan and exact recipe bytes are
retained before execution. A job ID binds its recipe, complete argv, working
directory, prerequisites, timeout and output roster. Source hashes bind the
controller and shared-lock helper; source changes require a separately declared
queue rather than silently changing a running experiment. The controller is a
local cooperative executor, not a sandbox or an authority to run unreviewed code.

`argv` is passed directly to `Popen(shell=False)`. Only arguments exactly equal
to `{recipe}`, `{output_dir}` or `{job_dir}` are substituted. Shell expansions,
metacharacters and command substitution remain literal arguments. The command
also receives `KWS_RESEARCH_JOB_ID`, `KWS_RESEARCH_RECIPE`,
`KWS_RESEARCH_RECIPE_SHA256`, `KWS_RESEARCH_OUTPUT_DIR` and
`KWS_RESEARCH_DEPENDENCIES`. The last names a hash-bound prerequisite receipt
and output manifest, not generation prompts or inferred labels.

## Complete, fail, resume

Each job receives one create-only `claim.json` **before** a worker or command
can start. An independent worker retains the job lock and heavy-model slot if
the queue-driving process dies. It writes separate stdout/stderr logs, command
identity, declared output hashes and `result.json` before the next ready job
starts. Output bytes and directories are synchronized before completion is
published. Exit zero without every declared regular output is a failure.

| State | Resume behavior |
| --- | --- |
| `pending` | Start when every prerequisite is completed, its output bytes still match, a slot is available and budget remains. |
| `running` | Leave the existing worker alone; another runner cannot claim it again. |
| `completed` | Verify the exact retained outputs and logs, then reuse; never recompute. |
| `failed` / `timed_out` | Preserve the failure; block dependents; unrelated ready jobs may continue within the remaining budget. |
| `unknown` | Preserve the unresolved claim, report incomplete cost accounting and stop further dispatch. Do not assume zero cost or retry. |

There are **zero automatic infrastructure or quality retries**. A single claim
is consumed even if the process dies before the command starts. This stays
within the plan's maximum two infrastructure attempts; it does not implement a
new automatic second-attempt policy. A new attempt needs explicit review and a
new declaration. Existing claims, receipts and artifacts are never reset.

`run` returns 0 for a completed queue or a successful bounded pause,
1 for failure/blocked/unknown outcomes,
2 for invalid identities or artifacts, 3 for exhausted or exceeded budget,
and 75 for an active controller/worker or occupied heavy-model slot. `status`
reads current evidence without starting work. Reinvoking `run` persists an
orphaned claim as `unknown`; live locks protect against premature recovery.

## Pause after preparation

For a new queue containing preparation followed by training, the optional
positive integer `--max-jobs N` limits new claims in one invocation. For example:

```sh
python3.12 -I -S -B tools/research_queue.py run \
  --root /private/research/queue-001 --max-jobs 1
```

After preparation completes, the controller returns `status=incomplete` and
`stop_reason=max_jobs_reached`, releases the shared model slot, and leaves the
next job without a lock, claim, worker, command or output. The caller can now
verify the actual saved vocabulary and initial float states. A later ordinary
`run` advances the same frozen request with the same remaining budget:

```sh
python3.12 -I -S -B tools/research_queue.py run --root /private/research/queue-001
```

Completed jobs are verified and reused; they do not consume the per-invocation
limit. Actual elapsed nanoseconds remain cumulative across invocations. A
failed newly claimed job still returns nonzero, and unresolved work or exhausted
budget takes precedence over the optional pause. Omitting `--max-jobs` retains
the existing completion-driven behavior. A zero exit code at a bounded pause
does not mean the entire queue is complete; inspect the returned status.

The controller identity remains strict. Existing queues must be read and
resumed with their original frozen source snapshot. This option belongs to a
newly declared queue using the new controller; it does not migrate historical
queues, rewrite receipts or bypass their source checks.

## Actual minutes and process lifetime

Each worker is given the smaller of its declared timeout and the remaining
aggregate budget. Its monotonic deadline starts at the durable dispatch claim,
including worker startup. Command timeout kills its complete process group.
The parent controller has a bounded extra recovery window if its worker fails
to return. Receipts measure actual monotonic nanoseconds through command
execution, cleanup and output/log hashing. Known actual durations are summed
once from terminal receipts, including failed and timed-out jobs; waiting for
an external slot is not billed as a started job.

Kernel scheduling, termination and persistence can finish after the command
deadline. Their observed time is retained as `deadline_overrun_nanoseconds`,
and aggregate overspend is explicit as `budget_overrun_nanoseconds`; elapsed
time is never clipped to the requested limit. An unknown outcome has no invented
duration and makes `actual_total_is_complete=false`, which blocks further jobs.

## Share the actual model slot

Direct and supervised entrypoints use:

```python
from research_job_lock import heavy_model_slot

with heavy_model_slot(heavy_lock_path) as lease:
    process = subprocess.Popen(
        argv,
        env=lease.child_environment(),
        pass_fds=lease.pass_fds,
        start_new_session=not lease.supervised,
    )
```

`KWS_RESEARCH_HEAVY_LOCK_FD` and `KWS_RESEARCH_HEAVY_LOCK_PATH` must appear
together. The inherited descriptor must refer to the same regular file with
one filesystem link and hold the exclusive flock. Closing a child context never unlocks its
parent's open file description. Stale/mismatched inheritance fails explicitly.

The queue additionally supplies `KWS_RESEARCH_SUPERVISED_JOB=1`. It is accepted
only with a verified inherited slot, exposed as `lease.supervised`. Supervised
entrypoints **must keep their child stages in the outer process group**; they
cannot daemonize, create escaping sessions or launch unmanaged descendants.
The maintained D0 trainer now propagates the descriptor through `run → worker`
and respects this group rule. Its stage workers have no child processes; a
supervised per-stage timeout kills that worker itself, while a standalone run
can terminate the stage's separate process group. This changes future launches,
not previously recorded D0 training code or checkpoints.

## Actual deployment gap

The GitHub hourly watchdog is **not deployed**. No new Actions schedule,
paid interface or remote worker has been enabled. The hosting entrypoint below
is available, but committing it or a service template does not install a live
service. A host loss, simultaneous controller/worker death, or an entrypoint
violating the group/descriptor contract can leave ambiguous or still-live work
requiring inspection. Surviving inherited locks prevent new cooperating model
work; missing terminal receipts never become success. A heartbeat establishes
local process liveness, not remote recovery or measured model progress.

## Host one declared queue continuously

`tools/research_queue_supervisor.py` hosts the existing controller. It does not
build job requests, discover other queues, select models, change budgets or
retry model work. Attach it only to the intended, already prepared declaration;
do not scan old queue roots and restart superseded experiments. An intentional
`--max-jobs` preparation pause must be reviewed before attaching the supervisor,
because ordinary queue execution advances all ready jobs.

```sh
python3.12 -I -S -B tools/research_queue_supervisor.py run \
  --root /private/research/queue-001 \
  --state-dir /private/research/queue-001-supervisor
```

The state directory must be separate from the queue and either new or bound to
that exact queue and supervisor source. Keep the matching `research_queue.py`,
`research_job_lock.py` and supervisor in a retained source snapshot; a running
checkout must not change beneath them. The queue's existing source validation
is unchanged. A source change needs its own reviewed invocation/state, never a
rewrite of the old binding or queue receipts.

Normal job completion immediately advances dependencies inside `research_queue`.
When an existing controller, live worker or another queue owns the shared model
slot, the supervisor waits and emits a heartbeat. It checks locks once per
second by default; it does not rehash all completed model artifacts every second.
The controller still validates source, prior outputs and actual remaining job
budget before dispatch. A lock race is handled by its existing exit code 75.

The replaceable `heartbeat.json` is written atomically every five seconds and
on state changes. It records supervisor identity, controller PID and the last
controller report, with an explicit observation time. That report is a snapshot,
not a live validation of artifacts. Per-invocation controller logs and terminal
results are retained under `sessions/`; immutable queue receipts remain in the
queue. A second supervisor using the same state directory exits 75 without
overwriting the active heartbeat.

```sh
python3.12 -I -S -B tools/research_queue_supervisor.py status \
  --state-dir /private/research/queue-001-supervisor
```

`status` reports lock-based liveness and heartbeat age without starting a model
or queue. An inactive process with an old `controller_running` heartbeat is not
a live supervisor. Read the actual queue with its retained controller when
checking artifact integrity or deciding how to handle missing evidence.

### Stop and restart behavior

- SIGTERM/SIGINT stops this supervisor's owned controller from dispatching
  more jobs. It does not signal the worker's separate group. The already started
  worker retains its original deadline, lock and cost accounting. It may finish
  after the supervisor exits; a later restart waits for it before advancing.
- If the supervisor is killed unexpectedly, its controller may survive. A
  restarted supervisor observes the existing lock and waits, rather than
  launching overlapping work. One unexpected controller-process recovery is
  allowed per invocation; the original queue decides whether any job may run.
- Completed outputs are reused. `unknown`, failed/blocked, invalid declarations
  and exhausted/overrun budgets terminate supervision. No model outcome is reset
  and no new attempt is fabricated. Host reboot can remove both controller and
  worker; an unclosed claim then remains `unknown`, requiring evidence-based
  recovery instead of an automatic repeat.

Exit codes are 0 for completed, 1 for failed/blocked/unknown, 2 for invalid
input/source, 3 for budget exhaustion/overrun, 75 for an already active supervisor,
130 for SIGINT and 143 for SIGTERM. Queue runner minutes remain the sum of its
actual job receipts. Supervisor waiting time is separate host wall time; a
hosted CI runner can still charge for waiting, so use a persistent local worker
for this service rather than parking a billed Actions runner.

### Optional user service

[`templates/kws-research-queue.service`](templates/kws-research-queue.service)
is an installation template for a host with a working systemd user manager.
Replace all `@...@` values with the retained interpreter, source and explicit
queue/state paths, then install it in the intended user's service directory.
Use the interpreter's required dynamic-library environment if applicable; for
the retained standalone Python build this includes its actual `lib` directory.
The example line is commented until filled with that verified path.

After installing the concrete unit, the operator can use
`systemctl --user daemon-reload` and
`systemctl --user enable --now kws-research-queue.service`. Verify the running
unit and a changing heartbeat before reporting deployment. Startup at boot or
after logout additionally depends on the host's user-manager policy. If the
host has no service manager, the foreground command works only while its host
process remains available; do not describe that as permanent hosting.

The unit uses `KillMode=process` deliberately: stopping the service addresses the
supervisor, which stops only its controller. The existing bounded worker must
retain its independent deadline and locks across a supervisor restart. It also
prevents automatic restart on terminal research outcomes; unexpected supervisor
death can restart, while unknown model work still cannot be retried.

The offline CI contracts use real, bounded, stdlib-only dummy commands. They
cover dependencies, successful reuse, failed exit, missing output, actual
budget exhaustion, two competing queues, controller death with surviving
worker, worker death with an unknown result, literal argv and the maintained
trainer's nested process-group boundary. They do not execute a model or claim
that earlier real experiments were scheduled by this controller.

`tests/test_research_queue_supervisor.py` additionally kills/restarts real
supervisor and worker subprocesses, verifies surviving-owner waits and exact
output reuse, and checks heartbeat ownership, unknown outcomes and unchanged
aggregate budgets. These tests do not install the user service.
