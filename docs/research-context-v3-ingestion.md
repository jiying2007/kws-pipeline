# Context v3 fixed ingestion smoke

`training/research_context_v3_consume.py` performs one bounded C1 consumption
check over the new context-v3 observations. Its purpose is to establish that the
verified complete transcripts, source families, loss weights and original
2601-symbol targets enter an actual optimizer loop. It does not establish that
C1 fits these examples, generalizes to new speakers, or passes a release gate.
Machine agreement remains a weak label and is never represented as human gold.

## Selection before observation

The immutable `kws-context-v3-ingestion-selection-v1` document is saved and
acknowledged before the first ASR claim in the 72-cell parent campaign. It
selects the first six lexicographically sorted declared training families and
all six declared development families. The 18 planned training source IDs and
18 planned development source IDs follow their original plan order. ASR results
and model predictions cannot select families or replace excluded observations.

Every one of the 72 planned cells receives an enrollment record. All observed
rows are replayed before selection. Missing audio stays explicitly unobserved;
uncertain, failed-native, OOV and infeasible rows retain their provenance and zero
loss weights. Training admission requires observed K1, observed K2 and ordinary
speech among the predeclared training rows. If those strata are unavailable,
preparation fails without substituting another family or launching a model.

An ordinary speech negative keeps every character of its complete observed
transcript as a nonempty CTC target. Keyword labels `[0,0]` represent confirmed
absence in that observed text. Unknown observations cannot become negatives.
The inherited vocabulary is not expanded or shortened to accommodate an ASR
result, and no transcript is shortened to the wake phrase.

All 18 planned development cells remain evaluation-only even when none is
eligible. Eligible development rows receive exactly one final inference-mode
readback after the optimizer has stopped. They receive no training draw, no
before-training diagnostic, and no hyperparameter or checkpoint selection.
A zero-eligible development cohort produces an explicit empty readback and no
invented metrics.

## Original algorithm and complete parent closure

The saved D1 preparation is checked with its unchanged original verifier in a
fresh isolated subprocess. The new consumer also requires exact equality of all
19 original algorithm, model, frontend, loss and queue source hashes. The
runtime manifest may differ only in its `validation` evidence metadata; its
Python and dependency contracts must remain identical. Both old and current
full source identities remain retained.

The original zero-step C1 checkpoint, 2601 vocabulary bytes and original token
file are copied exactly. Seed 20261105, AdamW learning rate 0.001, weight decay
0.01, gradient clipping and the weighted full-text CTC implementation are reused.
The final checkpoint audit verifies actual saved AdamW groups, parameter order,
finite correctly shaped first/second moments, pinned defaults and step 25 for
every parameter. Model and optimizer states are retained, not reconstructed.

Before admission, the frozen parent campaign validator runs `status` under its
own declared audio-preparation interpreter. All nine chunks must be complete,
with no active driver, unresolved/failed jobs or budget overrun. Every original
chunk completion receipt and its actual save acknowledgement is authenticated.
The consumer derives its complete package roster from those closures; an
operator cannot pass a preferred subset. The complete native/ASR weak views are
then replayed with their exact producer files, including cross-split and original
D1 source/PCM identity checks.

## Fixed budgets and artifacts

| Item | Fixed value |
| --- | --- |
| Candidate | Fresh original `c1-ctc2601` initialization |
| Planned train / development | 18 / 18 cells in six / six families |
| Optimization | 25 steps, batch four, 100 cyclic eligible train draws |
| Train observations | One before and one after per eligible train row |
| Development observations | One final readback per eligible development row |
| Worker | 120-second monotonic and signal deadline |
| Durable queue | One job, ten runner minutes |
| Shared-slot waiting | At most 60 seconds in the same live owner |
| Automatic retries | Zero |
| ASR/TTS calls | Zero |

The 100 training forwards and all diagnostic forwards are counted separately.
Each diagnostic retains its complete feature and logit tensors, complete target,
CTC NLL, greedy prediction and blank occupancy. High blank frame occupancy and
an empty decoded sequence remain separate measurements. The per-draw ledger
retains full text, target IDs, family identity, weights, raw normalized CTC loss,
weighted contribution and gradient norm.

`prepare` creates immutable audio, all-row projection, all-72 enrollment,
selection, original-source receipt, campaign closure and execution/data-source
copies. The completed output retains these bindings, initial/final checkpoints,
full consumption ledger, train/development row lists, feature observations and
saved numerical evidence. `audit` checks those retained values without any new
model construction, model forward or optimizer step.

## Prepare and execute

The preparation request has schema `kws-context-v3-ingestion-preparation-v1` and
exact keys `schema`, `anchor`, `anchor_manifest_sha256`, `view_loader`, `plan`,
`selection`, `selection_checkpoint`, `campaign` and `campaign_validator`.
`plan`, `selection`, `campaign` and `campaign_validator` are absolute-path pins
with actual `path`, `bytes` and `sha256`. The validator pin names the original
frozen `tools/research_context_v3_campaign.py`, not a later publication copy.
The selection checkpoint is its original externally acknowledged folder.

### Explicit metadata handoff admission

An additive request variant has schema
`kws-context-v3-handoff-ingestion-preparation-v1`. It keeps the same anchor,
plan, pre-ASR selection, original campaign and original validator pins. It
replaces the single `view_loader` field with two explicit absolute-path pins:
`handoff` and `handoff_validator`. The latter must name the exact frozen
`tools/research_context_v3_metadata_handoff.py` that owns that handoff.
The original request form and its frozen execution sources remain unchanged.

The new admission runs only the pinned handoff's `status --handoff` in the
declared status interpreter. It requires all four original closures and all
five continuation closures, their exact acknowledged chunk scopes, and the
acknowledged final handoff completion. The original campaign identity,
generation stage, canonical shared lock, already-consumed model calls and both
the 180-minute cumulative and 90-minute continuation accounting are preserved.
An active driver, partial closure, failed/unknown claim, missing result or any
overrun prevents training preparation. There is no generation or ASR retry.

The package roster comes directly from those authenticated closures. Its first
four packages use the original 14-file reader. Each later observed package uses
the additive 15-file metadata reader. Missing native returns stay explicitly
unobserved. The mixed dispatcher replays each package in a fresh process with
its actual producer source map, keeping each row's own consumer and adapter
schema. The metadata relation retains both original container hashes and the
exact array/PCM identity; complete observed targets and zero-weight exclusions
are unchanged.

Preparation retains both complete byte-identical reader snapshots under
`data-sources/context-v3-r1` and `data-sources/context-v3-metadata-r2`, alongside
the original declarations and source locations. Readback loads those retained
copies and checks the same package reports. The only permitted report
difference is that explicit loader location; the source hashes, revision,
adapter, projected rows, original containers and every other report field must
match. No original package or reader is rewritten. The fixed family selection,
19 original implementation sources, fresh C1 initialization, 25 optimizer
steps, 100 train draws and final development-only readback are identical.

Use the exact installed core interpreter and preserve its required shared
library environment. Preparation and declaration make no model calls:

```sh
python3.12 -I -B training/research_context_v3_consume.py prepare \
  --request /private/context-v3-ingestion/preparation-request.json \
  --output /private/context-v3-ingestion/prepared

python3.12 -I -B training/research_context_v3_consume.py declare \
  --prepared /private/context-v3-ingestion/prepared \
  --prepared-manifest-sha256 ACTUAL_PREPARATION_MANIFEST_SHA256 \
  --output /private/context-v3-ingestion/recipe.json

python3.12 -I -B training/research_context_v3_consume.py stage \
  --recipe /private/context-v3-ingestion/recipe.json \
  --output /private/context-v3-ingestion/stage \
  --active-plan configs/research/active-plan-20261105.json \
  --heavy-lock /private/parent-asr-runtime/heavy-model.lock

python3.12 -I -B training/research_context_v3_consume.py execute \
  --stage /private/context-v3-ingestion/stage/stage.json
```

`stage` verifies the canonical shared lock against the parent campaign and creates
one exact `context-v3-ingestion` queue job. Its arguments use the queue's actual
`{recipe}` and `{output_dir}` substitutions. The stage and recipe are reviewable
before the execution owner is started.

The owner saves `checkpoints/before-training` and waits for a real external save
acknowledgement. The worker verifies the same live owner plus exact
stage/queue/recipe/output/lock scope before any model construction. Temporary
slot contention with no claim keeps the same Gate owner; it never creates a
new inference attempt. An acknowledged but unused intent belongs to that live
invocation and cannot be reused after a restart.

After the queue ends, `checkpoints/training-terminal` saves the authentic queue
result and every partial or completed artifact before numerical readback.
Failed, timed-out and reconciled unknown claims are sealed without a retry.
A successful model job is audited in a fresh process; its readback or explicit
readback failure is then saved under `checkpoints/readback-complete`. A readback
failure does not erase the completed optimizer evidence or authorize training
again. Actual capsules and provider receipts are required; local ZIP creation
alone is not durable storage proof.

The owner retains full current-job evidence, fixed preparation, and parent
capsule descriptors/acknowledgements/provider receipts. It does not recursively
copy the nine earlier audio capsules into the new capsule.

## Offline validation

The focused suite uses fictional projected rows, tensor-only AdamW states and
real dummy queue processes with explicitly labelled fixture save receipts. It
checks full ordinary targets, immutable families and budgets, all-row exclusions,
zero development optimization, exact queue substitutions, live-owner binding,
unknown terminal closure, slot contention and no replay of old intents.

```sh
python3 -I -S -B tests/test_research_context_v3_consume.py
python3 -I -S -B -O tests/test_research_context_v3_consume.py
python3 -I -S -B -OO tests/test_research_context_v3_consume.py
python3 -I -S -B tests/test_research_context_v3_handoff_consume.py
python3 -I -S -B -O tests/test_research_context_v3_handoff_consume.py
python3 -I -S -B -OO tests/test_research_context_v3_handoff_consume.py
python3.12 -I -B tests/test_research_context_v3_consume.py --saved-tensors \
  OriginalIdentityTests.test_saved_adamw_steps_defaults_parameter_map_and_moments
```

The optional last check constructs only small tensors, never a model or an
optimizer. Passing these contracts is separate from the recorded real training
result and from the earlier tiny-fit outcome.
