# Bounded continuation of the frozen D1 generation plan

`tools/active_tts_expand.py` prepares a new continuation stage from the retained
first-eight recipe and its completed queue receipt. It keeps the complete
256-cell producer plan, source/reference bytes, content-check modes, model
assets, package sources, sampling parameters and `max_new_tokens=256` unchanged.
Preparation, status and export use the standard library; they do not load or
execute a model. Generated speech and transcripts stay in private artifacts.

## Scope and budgets

The first continuation targets `plan.cells[:64]`. The original eight must all
verify against an explicitly supplied terminal SHA-256. They are imported as
references to the original files. The next 56 are two ordered 28-cell jobs:
`plan.cells[8:36]` and `plan.cells[36:64]`. Each job has a 600-second hard queue
limit and the stage has a 1,200-second cumulative queue budget. The second job
depends on the first job's completed receipt and declared output hashes.

The entry point checks the actual queue request, frozen recipes, claim budget,
prerequisite receipts and cumulative measured duration before loading a model.
It inherits the verified heavy-model file descriptor and stays in the queue's
process group. The existing native network filter is installed before the first
ML import. The same initialization-time telemetry suppression, local asset
hash checks, CPU float32 configuration and four-thread limit apply.

History is separate from the new stage's allowance. `history.json` pins the old
cross-attempt budget and completed historical receipts. The old unknown duration
remains `null`, with its original conservative reservation. Status reports known
historical nanoseconds and known new-stage nanoseconds separately; neither is
presented as a complete lifetime total. An unknown claim does not become a
zero-duration job. No automatic retries are introduced.

For a new plan whose first-eight job completed successfully, preparation accepts
`--completed-base-history` instead of `--prior-budget`. It authenticates the
same explicitly pinned base result and its original durable claim, then records
the result's `actual_elapsed_ns` as the duration of that first-eight job only.
It does not invent an old failed attempt, substitute model-worker or outer-driver
timing, or claim a complete project/runtime lifetime total. This separate
`active-tts-completed-base-history-v1` is re-derived from the original receipts
whenever the stage is loaded. The existing unknown-history format and checks
remain unchanged; its unknown duration stays null and its reservation remains
separate. Both modes are validated before any new ledger or preparation files
are published, and neither adds time to the new stage's fixed 1,200-second
allowance.

A separately named infrastructure recovery is allowed only for one proved
pre-model failure: the first job has a known failed terminal, every old output
directory is empty, no selected cell has a claim or terminal, and later jobs
were never claimed. The first stage, execution source, logs and receipt hashes
remain immutable. `infrastructure_attempt=2` keeps the original
`stage_budget_minutes=20` but fixes the new queue to `queue_budget_minutes=19.99`
(1,199.4 seconds). Its `recovery` object pins the prior actual duration and
rejects a combined maximum above 1,200 seconds. Arbitrary reduced budgets or a
third infrastructure attempt are not accepted. Preparation also verifies that
the shared heavy slot can be acquired. Later status checks retain the original
failed attempt and add its measured duration to the new queue's known duration.
The global ledger's create-only `recoveries/<prior-stage-sha>.json` binds that
one recovery to its new stage hash, so renaming the stage cannot allocate a
second recovery against the same failed receipt.

The queue and producer use distinct canonical encodings. Queue job-definition
and dependency hashes include the trailing newline used by
`research_queue.canonical`. Producer plan and cell-identity hashes retain their
existing encoding without that newline. The regression suite executes the real
queue CLI with two standard-library probe commands, so its claim and dependency
hashes are produced by the controller itself rather than mirrored fixtures.

## Durable identity and recovery

All stages for this plan use one ledger beneath the pinned runtime:

| Path | Meaning |
| --- | --- |
| `plan-ledgers/<canonical-plan-sha>/binding.json` | Plan, producer, runtime, model and generation identity; first-eight import hash |
| `.../base.json` | Exact verified first-eight records and original terminal hash |
| `.../claims/<cell-id>.json` | Create-only consumption claim, attempt 1, owning stage/job/recipe |
| `.../cells/<cell-id>.json` | Create-only terminal outcome and retained-file hashes |
| `.../writer.lock` | Shared stage preparation/execution lock |

A cell's identity hashes the complete binding, complete producer cell and
complete reference record. The ledger path follows the canonical plan digest,
so using a different stage directory cannot reset a cell's history. Publication
uses a flushed temporary file and an exclusive hard link. Racing writers cannot
both enter the generation callback.

Before a callback, the global claim is durable. After it, all retained media and
receipts must validate before a terminal is published. The following rules are
deliberate:

- A terminal with valid claim and matching native bytes can be reused with zero
  generation calls.
- A claim without a terminal blocks that cell and any stage trying to consume
  it. Partial files remain available for investigation; no inference is retried.
- A terminal with missing or changed raw files is rejected, without generating
  a replacement.
- A technical quality failure with a saved raw return is a terminal
  `FAILED_NO_RETRY`. No gain adjustment, replacement or quality-driven rerun is
  performed.
- The special first-eight origin must exactly match the verified baseline
  record, have zero new calls and have no extension claim. Changing an origin
  string cannot bypass claim validation.
- A job invocation with no completed queue receipt cannot be restarted through
  this entry point. A completed cache must validate its queue output pins,
  inventory, outcome copies and global record hashes before reuse.

Further stages can request a later prefix using `--target-count`, up to 256.
Only missing cells are selected. Each new stage still allows at most 56 new
cells, in at most two jobs of at most 28. A prefix containing an unknown consumed
cell is rejected. Completed technical failures count as consumed cells.

## Per-job artifacts and the data interface

The existing producer receipt schemas are preserved exactly. Each job has its
own `outputs/generation` directory with the same complete-plan lock, and only
the cells generated by that job. Successful cells retain `started.json`,
`completed.json`, `native-f32.wav` and `audio-16k.wav`. Failed cells retain the
original `failed.json`; they never receive a fabricated completion receipt.

For model returns, `outputs/native/<cell-id>/` retains `call.json`,
`returned-array.npy`, `native.json` and `returned.json`. A one-dimensional float
return at 24 kHz also has `returned-native.wav`. The exact returned array and
native FLOAT WAV are written **before** the technical gate. An abnormal array
shape, dtype or rate retains its serializable array with `wav: null` and
`wav_unavailable_reason`, and becomes `FAILED_NO_RETRY` with a null
`record.native_wav`. An inference, disk or process failure can leave a consumed
unknown claim without complete raw evidence; it is not promoted to a technical
failure with invented audio.

`outcomes.json` is an immutable consumer handoff:

```json
{
  "schema": "active-tts-stage-outcomes-v1",
  "stage": "/private/stage/stage.json",
  "stage_sha256": "<SHA-256>",
  "job_id": "chunk-001",
  "job_recipe_sha256": "<SHA-256>",
  "plan_canonical_sha256": "<SHA-256>",
  "binding": {"generation": "<same fixed configuration; other identity fields retained>"},
  "selected_cells": ["<cell-id>"],
  "new_call_ids": ["<cell-id>"],
  "reused_cell_ids": [],
  "rows": [{
    "cell_id": "<cell-id>",
    "ledger_record_sha256": "<SHA-256 of canonical record plus newline>",
    "record": {"schema": "active-tts-plan-cell-v1"},
    "claim": {"schema": "active-tts-plan-call-claim-v1"}
  }]
}
```

The abbreviated `record` above is the **complete** terminal record in an actual
file: identity hash, origin, status, claim hash, `output_root`, `generation_root`,
relative `native_wav`, every retained `files[path]={bytes,sha256}`, technical
failure reason, call count and admission flag. `claim` is the complete original
claim, whose canonical bytes plus newline match the record's `claim_sha256`.
The imported first-eight origin has a null claim if it appears in a reused
handoff. Both the summary and the completed queue receipt pin these artifacts.
A data consumer can copy this evidence into a portable package without reading
the mutable global ledger.

`export-asr` requires every selected job to have completed and the target prefix
to have no pending or unknown cell. It creates deterministic, native-only ZIPs,
each containing at most 64 entries (normally 28):
`cells/<cell-id>/native-f32.wav`. Before adding a member, the exporter independently
parses its RIFF chunks and checks mono IEEE FLOAT32 at 24 kHz, 1–720,000 frames,
finite samples and a wide absolute numeric bound of 1,000,000 to avoid numerical
overflow. These checks establish decoder input eligibility; they do not admit
a sample for training. A decodable technical failure uses its saved
`native/<cell-id>/returned-native.wav` bytes for that archive member; the index
explicitly marks it as failed. The existing success-only weak-view adapter must
not interpret such a cell as completed; an explicit failed-source adapter is
required, with uncertain labels and disabled training masks.

Nonfinite, malformed, overlong or missing WAV returns stay in `excluded_cells`
with their complete terminal record, hashes, reason and outcome pointer. They
have no ASR input or invented text. They remain new consumed cells, not reused
cells. `new_generation_cell_count`, `asr_eligible_new_cell_count` and
`asr_excluded_new_cell_count` make that distinction explicit. An all-excluded job
keeps its outcome pointer under `job_outcomes` and produces no empty ASR ZIP.
Finite quiet, clipped or otherwise loudness-failed audio remains eligible if it
meets the decoder input checks; no sample values are repaired.

Each index partition names its precise `cell_ids`, archive hash, generation
root, outcome hash and job-recipe hash. The roster contains **only new calls
owned by that job**. All original or concurrently completed reused cells are
listed separately under `reused_cells_without_new_asr_request` and
`reused_bindings`. They do not enter new ASR ZIPs. The independent paired-mode
experiment is never part of this plan or these archives. No intended text is
placed in a decoder archive, and the extension entry point executes no ASR.

## Operation

Supply the existing private paths and their independently checked hashes to
`prepare`. It creates a new package and queue request, but neither initializes
nor runs the queue. Its `stage.json` and selected cells are reviewable before
execution. After review, the established research controller initializes the
queue from `queue-request.json` using the same heavy lock, and runs it. Model
jobs use the pinned TTS interpreter; preparation and status can use the pinned
core interpreter without ML packages.

```bash
python3.12 -I -S -B tools/active_tts_expand.py prepare --help
python3.12 -I -S -B tools/active_tts_expand.py status --stage /private/stage/stage.json
python3.12 -I -S -B tools/active_tts_expand.py export-asr --stage /private/stage/stage.json
```

For the completed-base mode, the exact preparation command takes the following
form. All hashes refer to actual retained originals; this command prepares
metadata and the shared plan ledger without initializing a queue or loading a
model:

```bash
python3.12 -I -S -B tools/active_tts_expand.py prepare \
  --base-recipe /private/first8/prepared/recipe.json \
  --base-recipe-sha256 <actual-recipe-sha256> \
  --base-job /private/first8/queue/jobs/first-eight \
  --base-result-sha256 <actual-result-sha256> \
  --completed-base-history \
  --target-count 64 --stage-id next64 \
  --output /private/next64/prepared \
  --heavy-lock /private/runtime/heavy-model.lock
```

The first64 prefix is not a new stopping rule. Later fixed prefixes up to256
remain available through the same entry point, subject to its existing per-stage
56-call limit, actual terminal history, and the next research decision. Existing
completed and consumed-unknown cells never become new calls merely because a
different stage name or prefix is supplied.

`tests/test_active_tts_expand.py` verifies the cell and budget boundaries with
small standard-library protocol fixtures, including real forked claim races.
These tests verify persistence and handoff behavior, not model quality or a
successful new generation run.
