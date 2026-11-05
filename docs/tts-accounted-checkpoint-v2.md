# TTS continuation after a consumed, unreturned invocation

This is a new declaration and a new queue. The original prefix-232 queue stays
`timed_out`. Its 21 returned cell records, one consumed claim without a return,
and all original files remain unchanged. Only the 58 cells at full-plan indices
198 through 255 may be called by the continuation. There is one create-only
registration for this original stage, and no automatic recovery of this queue.

The fixed chunks are 12, 12, 10, 12 and 12. The first three jobs account for
prefix 232. The last two require a separately sealed ASR-232 handoff. The queue
budget is 29.9 minutes; the previous 600,186,647,856 ns plus that allocation is
2,394,186,647,856 ns, below the combined 40 minute allocation. Actual receipts
are summed without clamping or treating unknown durations as zero.

The TTS declaration freezes the ASR control path, the seven data-source paths,
the `verify_accounted_seal` API, and the handoff rules. It does not guess future
ASR files or data-source hashes. After the real ASR-232 queue completes,
`seal-asr-gate` copies exactly the seven files from that control's
`source-snapshot/data` into a create-only retained source directory. Every new
weak view must report those exact seven producer hashes. A fresh isolated,
bounded, same-process-group Python child executes the full source/native/
request/queue/weak-view verifier. The gate records the actual source pins and
local/public/tree mapping. Each of the last two TTS jobs rechecks the retained
and control snapshots and executes that verifier before any cell claim. There
is no fallback to a changing data checkout. The first three TTS jobs do not
load the data verifier.

## Typed checkpoint

All `pin` values have exactly `path` (absolute), `bytes`, and `sha256`. JSON
record/claim hashes use canonical JSON plus one LF, as their original files do.
The existing full-plan and cell-identity canonical hashes have no trailing LF.

`checkpoint.json` has these fields:

```json
{
  "schema": "active-tts-accounted-checkpoint-v2",
  "declaration": "pin",
  "registration": "pin",
  "protocol_declaration": "pin",
  "prior_evidence": "pin of the unchanged sealed 176 evidence specification",
  "old_partial_proof": "pin",
  "plan": "pin",
  "plan_canonical_sha256": "sha256",
  "binding": "the unchanged original generation binding",
  "prefix_count": 232,
  "prior_count": 176,
  "planned_cell_ids": ["the exact full-plan prefix"],
  "prior_cell_ids": ["the exact first 176 IDs"],
  "owners": {"owner-id": "owner object below"},
  "returned_rows": ["returned row below"],
  "unobservable_consumed": ["unknown row below"],
  "eligible_cell_ids": ["returned, decodable IDs in plan order"],
  "excluded_native_rows": [{"cell_id": "id", "reason": "actual geometry reason", "ledger_record_sha256": "sha256"}],
  "accounted_cell_ids": ["the exact prefix IDs"],
  "budget": "actual cumulative queue accounting",
  "previous_checkpoint": "null at 232; pin of the unchanged 232 checkpoint at 256",
  "new_model_calls_during_checkpoint": 0,
  "training_admitted": false
}
```

At prefix 232 the returned rows are original indices 176..196 followed by new
indices 198..231 (55 rows if all new calls return). At prefix 256 they include
the additional 24 returned rows. The original 176 observations are carried by
`prior_evidence`, not regenerated or duplicated in these rows. Every returned
row has an eligibility decision; a technical quality failure with a legal
FLOAT WAV remains eligible. The one consumed unknown has no observation row.

An owner has exactly these fields:

```json
{
  "origin": "original-timed-out or continuation-completed",
  "job_id": "the actual job id",
  "stage": "pin of original stage or new declaration",
  "job_recipe": "pin of the actual frozen queue recipe",
  "queue_root": "absolute path",
  "queue_files": {"queue.json": "pin", "request.json": "pin", "active-plan.json": "pin"},
  "job_root": "absolute path",
  "job_files": {"relative filename": "bytes/sha256 record; exact recursive file roster including outputs"},
  "result": "pin of actual result.json",
  "outputs_root": "absolute path",
  "source_root": "absolute source checkout",
  "source_files": {"relative source filename": "bytes/sha256 record"}
}
```

The verifier checks the original controller's request/job/claim/dependencies,
terminal, logs and exact output roster. Original owners require the fixed
`timed_out` terminal and retained partial proof. New owners require real
`completed` queue terminals and complete inventories. A new completed owner
never authenticates a row belonging to the original timed-out job.

A returned row has exactly:

```json
{
  "cell_id": "id",
  "origin": "original-timed-out or continuation-completed",
  "owner_id": "key in owners",
  "ledger_record_sha256": "sha256 of original canonical-plus-LF record",
  "record_pin": "pin of original ledger cell file",
  "claim_pin": "pin of original create-only claim",
  "record": "complete original active-tts-plan-cell-v1 object",
  "claim": "complete original active-tts-plan-call-claim-v1 object",
  "eligibility": "original FLOAT24k decoder geometry result"
}
```

The consumed unknown has exactly:

```json
{
  "cell_id": "id",
  "origin": "original-timed-out",
  "status": "unobservable-consumed",
  "owner_id": "original-232-chunk-001",
  "identity_sha256": "original generation cell identity",
  "claim_pin": "pin",
  "claim": "original claim object",
  "output_root": "original outputs path",
  "artifacts": {"relative call/start filename": "bytes/sha256 record"},
  "absent_paths": ["absolute original cell terminal and all return/complete/failure paths"],
  "model_return_known": false,
  "may_retry": false,
  "asr_calls": 0,
  "native_system_calls": 0,
  "observed_text": null,
  "training_admitted": false
}
```

Its artifact roster must be exactly the original `call.json` and generation
`started.json`; all pinned absent paths and any added files in those two cell
directories cause verification to fail. No placeholder PCM or label is made.

## Export

`export/index.json`, schema `active-tts-accounted-asr-partitions-v2`, pins the
checkpoint and repeats its plan hash and prefix count. `partitions` contains
`{partition_id, archive: pin, cell_ids}`; exact native-only ZIP members are
`cells/<cell_id>/native-f32.wav`, with no extras. Eligible returned rows are
ordered by full-plan ID order and split into chunks of at most 28. Prefix 232
therefore exports 28+27 for the expected 55 legal WAVs. Final-prefix export
accepts the prior checkpoint pin and emits only the additional eligible cells.
`reused_cell_ids`, `excluded_native_rows`, and `unobservable_consumed_ids` are
explicit and disjoint from that stage's ZIP roster. The checkpoint still
retains every returned row and the original unknown evidence.

The `verify-checkpoint` command is pure stdlib and read-only. It returns the
verified accounted/returned/eligible/excluded/unknown ID sets and source pins;
it does not load numerical packages, an engine or a queue worker. `export`
first performs that verification and writes a new export directory once.
Budget accounting always reports actual nanoseconds and separate new-queue and
combined-allocation overruns. A checkpoint is an authenticated account of the
outputs, including a possible budget overrun; it is not a claim that an overrun
was authorized. No output is erased or elapsed time clamped to make it fit.
