# D2 runtime loss and persistent phase checkpoints

The 2026-11-05 archives retain four sealed D2 chunks: 48 original returns,
43 machine-weak candidates, five uncertain rows and 113 actual ASR calls.
The full drivers could continue after those snapshots. On 2026-11-06 their
local runtime and later receipts were absent. The remaining 1,744 declared
requests therefore have **post-checkpoint outcome unknown**. This does not
mean that all 1,744 actually ran, and it does not mean that they are pending.
Their elapsed time and call count remain unknown. The original records are
unchanged; no old request is reissued.

## Reconcile available evidence without reviving an old queue

`tools/research_d2_recovery.py` reads a create-only request:

```json
{
  "schema": "kws-d2-runtime-loss-request-v1",
  "local_runtime_lost": true,
  "post_checkpoint_execution_was_possible": true,
  "checkpoints": [
    {"root": "/private/restored-checkpoint", "manifest_sha256": "<independently pinned SHA-256>"}
  ]
}
```

The caller must obtain each manifest pin from the verified original archive,
not regenerate a manifest from whichever files survived. The reader checks
the exact member set, every file hash/size, conflicting copies of an original
path, campaign/stage/plan bindings, queue claims and results, selected-cell
rosters, native inventories and sealed ASR output pins. It resolves the old
absolute paths through archive metadata. It never edits those paths inside
an original JSON receipt, calls a model, or uses the live queue inspector to
classify missing archived claims as pending.

```sh
python3 -I -S -B tools/research_d2_recovery.py reconcile \
  --request /private/loss-request.json --output /private/loss.json
python3 -I -S -B tools/research_d2_recovery.py guard-plan \
  --plan /private/new-plan.json --loss /private/loss.json
```

The loss receipt retires the affected campaign scope and lists its proven
archived cells separately from post-checkpoint unknown cells. An explicit
new experiment must retain this history, have a distinct request identity,
and not claim to have completed the lost requests. Recovering authentic
original receipts can refine unknown outcomes; deleting an exclusion or
changing a batch name is not recovery. `guard-plan` rejects overlap by cell
ID and by the original semantic campaign identity, independent of JSON spacing.

## New campaign declarations require four external-save barriers

`research_d2_campaign.py declare` now creates `kws-d2-campaign-v2`. The same
generation, ASR, weak-view and bounded queue implementations still own model
work. The driver adds these barriers:

| Phase | Evidence saved before advancing | Next allowed operation |
| --- | --- | --- |
| `before-generation` | Exact campaign, plan, reference/source/model-lock metadata, selected job and cell IDs, source files, prior acknowledgements, immutable phase marker | One bounded generation job in the owning live driver invocation |
| `generation-complete` | Authenticated generation result, claim, native arrays/WAVs, failed returns, original cell receipts and ASR export | Prepare a separate ASR intent |
| `before-asr` | Exact audio-only inputs, source map, ASR recipe, initialized queue and source pins | One bounded ASR job in the owning live driver invocation |
| `chunk-complete` | ASR originals/audit, weak-view package and sealed chunk receipt | Consider the next generation chunk |

Known loss receipts are pinned with repeatable `--history-loss` arguments at
declaration and rechecked before execution. Historical v1 declarations are
not silently upgraded or granted new execution by the current driver. Their
frozen original sources and archived read-only evidence remain available.
Changing a schema string or source hash is not a handoff.

A barrier creates `checkpoints/<job>/<phase>/capsule.json` and a uniquely
named `checkpoint-<scope-digest>.zip`. The descriptor pins the exact ZIP size,
SHA-256 and internal member-manifest SHA-256. The saved ZIP includes its own
phase marker, so restoring its source files cannot erase the fact that an
intent existed. Source files are checked again after an intent is acknowledged
and immediately before the driver calls the existing model queue.

The driver waits without holding the heavy-model slot. By default one barrier
waits at most 900 seconds. Missing acknowledgement stops it; it does not run
a model and hope to save later. Existing per-job deadlines and cumulative
runner-minute limits remain unchanged. Waiting for storage is not reported
as model inference time.

## External-save receipt

The operator saves the **exact descriptor-named ZIP** through the persistent
artifact provider and retains its real finalized response. The upload tool
is external to the offline model worker. It then supplies:

```json
{
  "schema": "kws-persistent-checkpoint-save-v1",
  "status": "saved",
  "provider": "<actual provider>",
  "object_id": "<actual opaque saved-object ID>",
  "version": "<actual version, when returned>",
  "artifact": {"path": "<exact descriptor path>", "bytes": 123, "sha256": "<exact ZIP SHA-256>"},
  "member_manifest_sha256": "<descriptor manifest SHA-256>",
  "provider_response": {"actual finalized response": "retained without inventing success"}
}
```

The schematic example is not a valid upload receipt. The opaque ID must be
present in the actual provider response, and artifact identity must match the
descriptor. A failure response or a local boolean alone cannot acknowledge a
barrier. Acknowledgement is create-only:

```sh
python3 -I -S -B tools/research_d2_checkpoints.py acknowledge \
  --checkpoint /private/campaign/checkpoints/generate-001/before-generation \
  --save-receipt /private/actual-provider-save.json
```

If acknowledgement stopped after saving the provider response but before
writing `ack.json`, the same command can finish with the **identical original
receipt**. It verifies any existing acknowledgement and rejects a different
provider response. This recovers metadata only: it performs no upload, changes
no saved receipt, and does not make an old launch intent reusable.

The acknowledgement explicitly records that the provider response was
supplied by the operator. It does **not** claim an independently downloaded
remote-byte proof. The tool neither performs nor fabricates an upload. Save
receipts and original audio stay private; public tests use labeled synthetic
storage responses and launch no speech models.

## Crash and restore behavior

A launch intent belongs to one live driver invocation. A new invocation sees
an existing intent without an authentic completed queue result as ambiguous,
even if a restored local queue appears pending. It stops instead of granting
the old intent a second launch. Authentic completed generation can be reused
for a separate ASR phase; completed ASR can be reused to build the final weak
view. Failed, timed-out or consumed-unknown model jobs retain the existing
no-retry rule.

Download persistent capsules by their real provider identities and verify
their externally pinned hashes. Restore parent capsules in order and keep
the original paths. The restore command checks every ZIP member, path and
existing-file conflict before writing. It permits only explicit allowed
roots and never overwrites changed bytes or rewrites receipt paths:

```sh
python3 -I -S -B tools/research_d2_checkpoints.py restore \
  --capsule /private/downloaded-checkpoint.zip --sha256 '<saved ZIP SHA-256>' \
  --allowed-root /private/original-runtime \
  --allowed-root /private/original-source
```

Restore reconstructs the original descriptor and phase marker. It creates
no provider acknowledgement and grants no model launch. Existing saved
acknowledgements must themselves be restored from later capsules, or the
exact completed-phase capsule must be saved again and actually acknowledged.
Missing parent capsules/source/model assets remain a blocking dependency.
An old launch intent cannot be made reusable by uploading it again.

## Other bounded model experiments

The primitive is reusable without creating another model runner:

```python
from research_d2_checkpoints import Gate

gate = Gate()
gate.checkpoint(intent_folder,
                {"phase": "before-training", "queue": queue_pin,
                 "recipe": recipe_pin, "selected_ids": fixed_ids},
                exact_dependency_files, intent=True)
# Only now call the existing bounded queue. Authenticate its result first.
gate.checkpoint(completion_folder,
                {"phase": "training-complete", "queue": queue_pin,
                 "recipe": recipe_pin, "result": authenticated_result_pin},
                exact_result_and_dependency_files, intent=False)
```

The owning process must remain alive across acknowledgement. The generic
primitive does not infer which files make an experiment complete: its caller
must supply the actual queue/recipe, consumed inputs, execution-source pins
and authenticated outputs. It is evidence persistence, not a label review,
ASR correctness claim, model-quality gate or permanent compute host.
