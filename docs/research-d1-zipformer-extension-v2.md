# Extending native Zipformer observations from 64 to 256

`tools/run_d1_zipformer_extension.py` creates a new bounded job after the full
256-cell terminal closure and the actual observable-row preparation complete. Its implementation and tests
do not imply that future preparation or inference has happened.

The new execution tree uses the controller with bounded queue advancement.
The original 64-row controller and preparation source remain in their original
frozen tree. A fresh isolated stdlib subprocess verifies that old tree and its
results. The parent uses the new controller to verify the new preparation.
Neither side overrides a source-record function or relaxes its identity check.
The model, runtime configuration, native profile and decoder remain fixed.

## Required real bindings

The v2 recipe is a strict object with these fields:

| Field | Requirement |
| --- | --- |
| `schema` | `kws-d1-zipformer-extension-job-v2` |
| `completed_preparation` | Actual completed full preparation queue root, job ID, terminal SHA and MANIFEST SHA |
| `previous_native` | Actual completed original 64-row native queue root, job ID, terminal SHA and MANIFEST SHA |
| `previous_source_root` | Absolute path to the unchanged original execution source |
| `previous_source_files` | All 25 exact original source hashes, checked before loading old modules |
| `final_prior_evidence` | Actual final sealed prior-evidence path and raw-file SHA |
| `plan_sha256` | The original frozen 256-cell plan identity |
| `source_ids` | Sorted, unique, complete 256-cell roster |
| `held_out_source_family_ids` | The same two development families |
| `models`, `profile_sha256` | The same local model assets and fixed profile as the original run |
| `source_files` | `source_records()` from the newly frozen execution tree |
| `timeout_seconds` | Positive and at most 120 |

Each completed binding has exactly `queue_root`, `job_id`, `receipt_sha256` and
`manifest_sha256`. Unknown future hashes are not valid declarations. The new
job runs in a separate one-job queue with no fabricated cross-queue dependency,
a maximum two-minute global budget and the same heavy-model lock.

## Previous 64: immutable evidence, zero new calls

The old queue terminal, claim, logs and output records are checked first. All
25 old source pins must match the original source directory before any of its
modules load. A fresh `python -I -S -B` child imports the original entrypoint and
uses its original `audit_finished` and controller. It returns the verified
rows, observation objects, identities and actual receipt bindings. The child
stays in the parent's process group and has a timeout bounded by the parent's
remaining deadline. It imports no ML package and makes no model call. Old TTS
queue receipts needed for exclusions use the same isolated old-source reader.

Each old source ID must occur once in the full preparation. Its global plan
identity, original input WAV/PCM hash, sample count, source family, speaker,
split, complete projected label identity, keyword targets, masks and weights
must match the old row. Only the prepared audio file location may differ. Any
drift blocks execution; it is not repaired by relabeling or another model call.

The original output directory is retained under `reused/original-output` with
identical file bytes, including the old identity, claims, native results,
observation objects, summary, source files and MANIFEST. The original queue
records are retained separately. Copied evidence remains explicitly old
evidence; it is never assigned the new native identity.

## Full plan closure and explicit unobservable failures

The plan always has 256 cells. The observable count can be smaller. The final
`d1-sealed-prior-evidence-v1` declaration must have `prefix_count=256`, and its
view pins must equal the actual preparation's aggregate parents exactly.
Every view is replayed through its retained data loader before checking the
complete projected row identities against the sealed preparation.

The difference requires explicit hash-bound excluded exports. For each one,
the verifier checks the original frozen plan, stage, job recipe, actual
completed queue receipt, selected cell roster, immutable outcomes and complete
failed terminal with its create-only claim. Bulk plan and cell digests omit the
trailing LF; persisted terminal and claim hashes include it. The original
failed artifacts are checked against the terminal's file inventory. A terminal
cannot be silently omitted, and an excluded cell cannot also be an observed
row. Observed IDs plus excluded IDs must equal all 256 planned IDs exactly.

An export flag alone cannot make a waveform unobservable. The verifier repeats
the frozen decoder input contract: complete RIFF, mono 24 kHz FLOAT32, 1 through
720,000 frames, finite samples and absolute values at most 1,000,000. Original
file identity is checked before parsing. Finite quiet or clipped waveforms that
failed the producer's quality gate remain observable; attempting to exclude
them is rejected. A missing WAV or invalid decoder geometry keeps its exact
recomputed reason and original failed artifacts.

All original 64 must remain observable. Unobservable cells retain their reason,
original export, terminal and claim under `terminal-coverage.json` and
`unobservable.jsonl`, with zero calls and no invented text, PCM, keyword target
or loss. The corresponding small immutable evidence files are retained too.
Failure receipts and original arrays stay in the generation evidence packages.

## Additional observable cells: one new call each

Only the observable set difference, at most 192 cells, receives a new create-only claim and native call under
`new/claims` and `new/native`. The entrypoint reuses the fixed decoder and exact
PCM conversion. It adds no padding or gain changes and performs no threshold
search. Offline/telemetry environment and the native network guard precede ML
imports. The actual queue lease, process group and hard deadline remain in force.
If every additional cell is explicitly unobservable, the 64 original results
are still reused, the new-call ledger is empty, and NumPy and the native engine
are never loaded.
Consumed or partial claims are not retried.

The full `observations.jsonl` contains one explicit wrapper per observable row. Each records
`origin`, `new_native_call`, the actual native identity and file hash, and the
original observation object and hash. Old rows also cite their original queue
terminal. Thus 64 reused observations are distinguishable from at most 192 new calls
without modifying either underlying native record.

## Reports and readback

Counts are separated into `all_observable`, `previous64` and `new_observable`, each with training
and development summaries. The original 16 development rows remain separately
visible from at most 48 additional observed development rows. Unobservable
terminal counts are reported separately by split. The same per-keyword known-mask rule applies;
failed, uncertain or quarantined rows retain raw events but contribute no
negative-label or qualification denominator. This remains a machine-weak native
event diagnostic, with no shared CTC token score or quality winner.

The original queue elapsed time is retained as an original measurement, while
the new queue measures its own actual elapsed time. The entrypoint never
recasts copied native records as newly incurred calls.

`audit_finished` checks the new inventory, exact original retained bytes,
native claim/result identities, strict old label identity and all merged
observations and summaries without loading a model. The stdlib contract tests
use explicitly fictional native receipts and cover global joins, source/PCM/
label drift, immutable original bytes, old identity retention, missing evidence,
separate development counts and the import boundary.
