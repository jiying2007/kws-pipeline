# Complete an existing D2 stage

New declarations now use the mandatory per-phase persistence protocol in
[D2 durable recovery](D2_DURABLE_RECOVERY.md). They stop at each external-save
barrier until a real provider receipt acknowledges the exact capsule. A new
driver invocation cannot reuse an ambiguous old launch intent. The v1 sources
that executed the 2026-11-05 campaigns remain frozen; incomplete archived
history is reconciled read-only and cannot be restarted as pending work.

`tools/research_d2_campaign.py` connects the existing generation queue to the
existing audio preparation, selective ASR and D2 weak-view producer. It does not
declare a new TTS plan, retry a cell, import D1 work or change historical queues.

The driver completes one chunk in this order:

1. Verify or advance the declared generation queue by **one** new job.
2. Authenticate its completed queue receipt and native inventory; export the
   original FLOAT WAV returns with `research_d2_generate.export_job`.
3. Prepare audio-only ASR inputs and build the D2 source map.
4. Declare or verify the selective ASR request and its single-job bounded queue.
5. Verify the ASR final audit, create or replay the D2 weak view, and seal the
   chunk's artifact pins before considering another generation job.

No available native return produces no ASR job and no training row. Native
returns that failed generation quality checks remain available to ASR; the
existing weak-view policy decides their mask from retained evidence. Intended
TTS text is never promoted to an observed transcript by this driver.

## Fixed layout

The campaign root contains `campaign.json`, a replaceable `heartbeat.json`, and
`chunks/generate-NNN/`. Each chunk uses these paths:

| Relative path | Meaning |
| --- | --- |
| `export/source-original.zip`, `export/export.json` | Authenticated original return export |
| `inputs/inputs.json` | Opaque audio-only ASR inputs |
| `source-map.json` | Audio/source routing identities |
| `asr-request.json` | Frozen selective ASR request |
| `asr-queue-request.json`, `asr-queue/` | Existing durable queue formats |
| `asr-queue/jobs/asr-selective/outputs/asr/` | Native ASR evidence and final audit |
| `weak-view/` | Independently replayable machine-weak package |
| `complete.json` | Create-only verified chunk completion |

The ASR queue ID is `d2-generate-NNN-asr`; its job ID is `asr-selective`. Its two
declared outputs are `asr/audit/final-summary.json` and
`asr/audit/file-hashes.json`. The ASR command uses the queue's `{recipe}` frozen
copy. The selective request's output is the nested `outputs/asr` path above,
which remains absent until the ASR job starts.

These paths also allow an independently completed first chunk to be adopted.
Create the manual artifacts with the same budget, threads, policy, runtime,
queue definition and source-owner pins. The driver revalidates every completed
phase before reuse; it never guesses paths or silently replaces a declaration.
`declare` accepts an existing root containing only in-scope `chunks/` paths.

## Declare once, pause and continue

Use the existing Python 3.12.16 **ASR environment** to host the campaign. It has
the required NumPy, SciPy and soundfile packages; the current core environment
lacks soundfile. The declared generation jobs still use their frozen TTS Python.
Do not use `-S` for a real campaign. Declaration and restart verify the driver
interpreter/package identity before any model claim. Set the matching
interpreter's library path where needed.

```sh
env LD_LIBRARY_PATH=/absolute/python-runtime/lib \
  /absolute/runtime-asr/venv/bin/python -B tools/research_d2_campaign.py declare \
  --stage /absolute/existing-eight-001/stage.json \
  --output /absolute/existing-eight-campaign-001 \
  --data-root /absolute/kws-data \
  --asr-runtime /absolute/runtime-asr \
  --fsmn-tokens /absolute/fsmn/tokens.txt \
  --max-chunks 32 --total-minutes 480 \
  --asr-job-minutes 10 --asr-budget-seconds 570 --threads 4

env LD_LIBRARY_PATH=/absolute/python-runtime/lib \
  /absolute/runtime-asr/venv/bin/python -B tools/research_d2_campaign.py run \
  --campaign /absolute/existing-eight-campaign-001/campaign.json \
  --max-chunks-this-call 1

env LD_LIBRARY_PATH=/absolute/python-runtime/lib \
  /absolute/runtime-asr/venv/bin/python -B tools/research_d2_campaign.py run \
  --campaign /absolute/existing-eight-campaign-001/campaign.json
```

`--max-chunks` freezes the allowed generation prefix. The invocation option
`--max-chunks-this-call` bounds newly completed chunks; previously sealed chunks
are reverified and do not consume that invocation limit. Waiting for a live
worker preserves the same invocation limit. To continue, use the same campaign
file and root.

The stage directory receives one create-only `campaign-driver.json` registration
and one cooperative `campaign-driver.lock`. A second driver declaration for
that stage is refused. The full campaign pins its stage, queue identity,
pipeline/data producer files, policy and optional FSMN vocabulary. Editing those
inputs after declaration fails verification.

## Budgets, recovery and hosting

The campaign total is the sum of **actual queue job nanoseconds** for all
selected generation jobs and every registered chunk ASR job, including adopted
completed work. Unknown duration is explicitly unresolved, never counted as
zero available work. Before dispatching a generation job, the driver reserves
its current queue allowance plus a full downstream ASR job; before ASR, it
reserves the full ASR allowance. The original queue enforces the actual process
deadline. The campaign never widens a queue's frozen budget.

Generation and ASR must point at the same ASR runtime `heavy-model.lock`. A live
controller or model owner is waited out while a five-second heartbeat continues.
Only the existing queue creates model claims. After driver death, its bounded
queue worker can finish independently; restarting this same campaign first
checks that worker's authentic result. A failed, timed-out or unknown job stops
the campaign and is not submitted again. Partial non-model output directories
are preserved and rejected, not overwritten.

Exit codes are `0` for completed or an explicit chunk-limit pause, `1` for a
failed/unknown job, `2` for invalid or changed inputs, `3` for exhausted/exceeded
budget, and `75` if another driver owns the stage. `status --campaign ...`
checks registration, source pins, queue receipts and sealed artifact pins without
launching work.

For a persistent Linux host, `docs/templates/kws-d2-campaign.service` is an
optional user-service template. Substitute its absolute paths and inspect it
before installation. No service is installed or enabled by these tools. The
environment must keep the declared repositories, runtime libraries, models and
private artifact paths available for the entire campaign. Do not simultaneously
host the generation queue with a second controller that advances all its jobs;
the campaign owns the interleaving with ASR.

## Lightweight verification

`tests/test_research_d2_campaign.py` uses actual subprocess queues, claims,
receipts and budgets with explicitly synthetic audio/ASR/weak-view adapters.
It checks phase order, immutable binding, restart reuse, the invocation cap,
combined budget reservation, concurrent-driver exclusion, failed/unknown
outcomes, and partial/tampered artifacts. It never imports or starts a model.
