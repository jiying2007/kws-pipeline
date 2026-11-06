# Consume the completed acoustic observations

The training/research_acoustic_consume.py entry adds one separate consumer view
and a fixed C1 ingestion smoke. It reuses the original observer, full2601
vocabulary, zero-step C1 checkpoint, feature extraction, weighted CTC loop,
exposure ledger, saved-tensor scorer and local research queue. It does not
introduce a scheduler.

## What can be consumed

The frozen package contains 124 variants of 14 positive mother recordings.
Of its 84 inherited full-text CTC candidates, 60 have the training role and 24
have the development role. The remaining 40 are training-role recorded-noise
mixtures whose full-text CTC masks stay disabled in this experiment. No ordinary
negative, natural-negative hour, independent speaker or real device recording is
created by this adapter.

A fresh stdlib-only subprocess executes the exact frozen observer. It verifies
the input stage and source hashes, original variants and parent/asset/split/PCM
bindings, all 16 completed native jobs, existing native audits and observation
views, the complete 124-row / 372-recognition aggregate, and the final durable
acknowledgement. Every required output must already exist. Failed or unknown
terminal jobs are not a complete campaign. The original driver lock is opened
read-only and held nonblockingly during replay; an active campaign is not stolen.

The new projection retains all 124 rows, their inherited labels, original parent
rows, complete three-engine observations and quarantine reasons. A non-noise
row may contribute full-text CTC only when all three complete, unflagged
transcripts agree with the parent's whole text and unpadded CTC geometry is
feasible. Its confidence weight stays 0.5. Other rows have no consumer CTC loss;
the original row and its labels are not edited. Agreement never restores the 40
noise CTC masks. Development rows remain in their original families and never
enter the optimizer.

The fixed vocabulary is checked against its original token bytes. Reference
PCM/WAV, parent/source ancestry, speaker/family and acoustic-asset aliases are
checked across all rows, including quarantined rows, and the original D1 anchor.
Waveforms are copied byte for byte without normalization, trimming or a new RMS
admission rule.

## Fixed smoke and evidence

The smoke uses one fresh C1 initialization, 25 optimizer steps, four draws per
step and 100 exposures. The accepted training rows cycle in their saved original
order. It uses the unchanged full-target-length-normalized CTC times saved
confidence, averaged over batch items. The optimizer and clipping settings come
from the verified D1 anchor. This positive-only smoke demonstrates consumption;
it does not implement the D1 positive/ordinary/near-negative comparison.

Training has a 120-second worker deadline within one 10-minute queue job, the
canonical existing heavy-model lock, and no automatic retry. Saved evidence
includes every consumed row/PCM/label/weight binding, initial and final state,
optimizer moments, train before/after logits, and one development readback.
The audit reconstructs features from the retained PCM and recomputes metrics
from saved logits and optimizer tensors without constructing a model, running
a model forward, or updating an optimizer.

No fit pass, quality winner or generalization result is claimed by this smoke.
The development readback does not select or tune a new attempt. A subsequent
effect experiment needs a separately fixed recipe with verified ordinary
negative data and the existing source-isolation checks.

## Entry points

Use the pinned core Python 3.12.16 environment. A preparation request has exactly
these fields; every path and hash refers to existing evidence:

    {
      "schema": "kws-acoustic-consumption-preparation-v1",
      "stage": "/absolute/completed-acoustic/stage.json",
      "stage_sha256": "<exact stage file SHA256>",
      "observer": "/absolute/frozen-observer/tools/research_acoustic_asr.py",
      "observer_sha256": "<exact original observer file SHA256>",
      "anchor": "/absolute/original-d1-preparation",
      "anchor_manifest_sha256": "<exact original D1 manifest SHA256>"
    }

Preparation, verification and declaration do not import a model runtime:

    python -I -S -B training/research_acoustic_consume.py prepare \
      --request /absolute/consume-request.json --output /absolute/new-prepared
    python -I -S -B training/research_acoustic_consume.py verify \
      --prepared /absolute/new-prepared --manifest-sha256 PREPARED_MANIFEST_SHA
    python -I -S -B training/research_acoustic_consume.py declare \
      --prepared /absolute/new-prepared --manifest-sha256 PREPARED_MANIFEST_SHA \
      --output /absolute/consume-recipe.json
    python -I -S -B training/research_acoustic_consume.py queue-request \
      --recipe /absolute/consume-recipe.json --heavy-lock /absolute/canonical/heavy-model.lock

Keep recipe and queue files outside the frozen prepared directory. Save the
returned queue-request JSON and use the existing queue's init entry:

    python -I -S -B tools/research_queue.py init \
      --root /absolute/new-queue --request /absolute/queue-request.json \
      --plan configs/research/active-plan-20261105.json \
      --heavy-lock /absolute/canonical/heavy-model.lock

The owner durably saves prepared inputs and the launch declaration before
invoking the existing queue:

    python -I -S -B tools/research_queue.py run --root /absolute/new-queue --max-jobs 1

There is no second campaign driver. Direct run is rejected without the queue's
inherited lease and exact job/recipe/output environment. Failed or consumed jobs
stay terminal; creating another output directory is not a retry authorization.
After the owner saves the terminal output, use:

    python -I -B training/research_acoustic_consume.py audit \
      --output /absolute/new-queue/jobs/acoustic-positive-ingestion/outputs \
      --manifest-sha256 OUTPUT_MANIFEST_SHA

The default focused tests run in normal, -O and -OO stdlib-only modes.
They use explicitly invented metadata/audio and never launch a model.
The optional --saved-tensors mode also authenticates saved invented logits and
PCM features in the pinned core runtime. It rejects model construction, forward
calls and optimizer construction while checking hash and score counterexamples.
