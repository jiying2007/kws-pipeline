# Explicit context-v3 metadata continuation

`tools/research_context_v3_metadata_handoff.py` is a separate owner for one
authenticated, stopped campaign. It is not a generic resume switch. Admission
requires the exact four completed context chunks, forty TTS claims/returns,
seventy-one saved ASR calls, completed generation job five, and unclaimed
generation jobs six through nine. It also requires the original pure failure's
durable capsule and the original source/runtime identities.

The original reader equated two independently serialized FLOAT WAV containers.
The retained generation-five counterexample has identical sample bytes and a
one-second difference in the libsndfile PEAK timestamp. The additive metadata
reader separately authenticates each complete original file, the original
native array, and the exact narrow timestamp relation. It does not rewrite an
old WAV, substitute a PCM-only identity, or rerun generation five.

## Ownership and budget

The original campaign declaration, queue, recipes, generator, model workers,
four weak packages and forty consumed TTS calls stay unchanged. Only ASR for
chunk five and generation/ASR for chunks six through nine are eligible for new
claims. The continuation takes the original campaign-driver lock and the same
shared heavy-model lock. Original generation recipes/argv remain unchanged.
Its new guarded ASR entry checks the continuation's own exact durable intent
before delegating to the unchanged selective-ASR implementation.

The original aggregate limits remain 72 TTS calls, 216 ASR calls, and 180 actual
runner minutes. The continuation additionally permits at most 32 TTS calls,
120 ASR calls, and 90 new runner minutes. Every model queue has a ten-minute
limit and the existing 570-second worker budget. Its live owner has a
5,400-second launch allocation and reserves 600 seconds again after each save
acknowledgement. Persistence of a consumed terminal result is still allowed
after this launch deadline. There are no automatic retries.

The source declaration is create-only. A saved pending intent belongs to one
live owner, even if it has not produced a claim. A temporarily busy model slot
may be waited on for at most sixty seconds by that same owner. A restarted
owner cannot reuse its predecessor's intent. Failed, timed-out and reconciled
unknown queue receipts are saved before stopping. A KeyboardInterrupt after a
claim is classified as an interrupted model phase, not a zero-model derivation
failure.

Fresh declarations create all five empty chunk parent directories before the
original create-only exporter runs. The layout helper rejects an existing
chunk tree and does not alter an already declared execution. Its regression
uses authenticated temporary queue receipts and the original exporter to
reproduce the missing-parent failure, then verifies export and deterministic
readback after layout creation, with no model calls or changed queue results.
Because evidence capsules retain files rather than empty directory entries,
the pure export phase also recreates a missing empty chunk parent after a
restore. It rejects file and symlink collisions, preserves existing exports,
and leaves all model-intent ownership and no-retry checks unchanged.

## Exact readers and closures

Each original package uses `context-v3-r1` with its original fourteen source
files. Each new package uses `context-v3-metadata-r2` with the additive reader
and the unchanged original fourteen files. The mixed projection dispatches
each package in a fresh subprocess with its declared reader root and complete
source hashes. R1 rows retain their original schema; R2 rows explicitly retain
the metadata adapter schema and native container relation.

Requested text never becomes an observed transcript. Ordinary-speech negatives
retain their complete observed nonempty CTC targets. Failed native returns,
uncertain labels and unsupported targets retain zero objective weight. A chunk
with no ASR-eligible native return closes without an ASR claim or fabricated
weak row; its unobserved planned cells remain available to enrollment audits.

The new manifest is `handoff.json`, schema
`kws-context-v3-metadata-handoff-v1`. Derived chunks live under
`chunks/generate-005` through `chunks/generate-009`. Each completes generation
evidence, new ASR intent, ASR terminal evidence, exact-reader weak replay,
full2601 projection and a durable `chunk-complete` acknowledgement before the
next model phase. The old four completions and new five completions are bound
by final `complete.json` and `checkpoints/handoff-complete`.

`status --handoff PATH` authenticates all completed payloads and compares each
acknowledgement to the exact handoff/job/phase/completion scope. It reports the
combined model accounting, explicit package reader declarations, driver-lock
state, and final completion/checkpoint pins. Downstream ingestion must require
the completed final receipt, inactive driver, and this exact closure. It must
not replace the original campaign's data root.

## Commands and no-model checks

Run `declare` in the original campaign's recorded driver runtime, with explicit
`--campaign`, `--original-validator`, `--metadata-data-root`, `--failure-receipt`,
`--failure-checkpoint` and a new `--output` directory. Declaration performs no
model calls. Review that concrete source/budget declaration before invoking
`run --handoff PATH` with the durable checkpoint saver active.

`tests/test_research_context_v3_metadata_handoff.py` uses temporary no-model
queue/Gate fixtures for exact scope, source membership, owner/restart,
deadline, interrupted claim, zero-observation and import-boundary cases. The
optional retained-data case accepts `--retained-campaign` and
`--retained-validator`: it runs the unchanged original gen6 intent verifier
against authentic stage/recipe pins in an isolated temporary checkpoint,
checks all loaded dependency byte identities, and proves that the real gen6
intent and claim remain absent. Its temporary provider responses are explicitly
test fixtures and are never stored as actual experimental acknowledgements.
