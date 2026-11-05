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
paid interface or remote worker has been enabled. The local command does not
install an always-on service. A host loss, simultaneous controller/worker death,
or an entrypoint violating the group/descriptor contract can leave ambiguous
or still-live work requiring inspection. Surviving inherited locks prevent new
cooperating model work; missing terminal receipts never become success.
The watchdog gap cannot be represented as automatic remote recovery.

The offline CI contracts use real, bounded, stdlib-only dummy commands. They
cover dependencies, successful reuse, failed exit, missing output, actual
budget exhaustion, two competing queues, controller death with surviving
worker, worker death with an unknown result, literal argv and the maintained
trainer's nested process-group boundary. They do not execute a model or claim
that earlier real experiments were scheduled by this controller.
