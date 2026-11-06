# Fixed C1 old/new source allocation

This is one optional follow-up to the completed balanced first64 augmentation.
It asks whether a different allocation of the **same 8000 training forwards**
changes the saved development outcomes. The first step is a diagnostic of saved
posteriors. Preparing this candidate does not authorize a model invocation.

The new entry point is `training/research_c1_source_allocation.py`. Its only
training change is the fixed sampler `old3-new1-fair-cycle-v1`: every four-item
batch visits three original-cohort rows followed by one new-cohort row. Each
cohort is traversed in its existing frozen row order. Each row independently
rotates its visit offsets through 0, 200 and 500 ms.

## What is held fixed

The control is the actual completed augmentation, bound to its saved recipe
SHA256 `75b5743222f96dde75fca0191159cb74a4a53d55531e625d4a4d255b46f2b0cd`
and output manifest SHA256
`06dc3b01a63500a9a88f149cdc73ba50493849f620668f909354b856456c8ea3`.
Its exact checkpoint is
`13cdff7365f5c34e0e8f09d73f32d73d90b462a447245506d7acac20695d8f92`.
The control is reused; no baseline model is rerun.

The candidate retains the exact 96 eligible train rows (52 original and 44 new),
34 development rows, four new uncertain zero-loss rows, complete observed weak
transcripts, masks, original CTC weights, vocabulary and serialized zero-step
initialization. All 134 evaluation rows retain all three offsets, including
the four rows without full-text targets: 402 final observations. Development
and uncertain rows never enter optimizer draws.

The model, frontend, full-text CTC, known-added-zero blank term, AdamW settings,
batch size4, gradient clipping, seed, 2000 optimizer steps and 8000 forward
budget are unchanged. There are exactly 6000 old-source draws and 2000
new-source draws. Each old row receives115 or116 visits; each new eligible row
receives45 or46 visits. The original equal-row cycle remains available unchanged
and still yields4348 old plus3652 new draws on this roster.

The experiment changes the **entire fixed sampling policy**: cohort quotas,
within-batch old/new composition, and the resulting row-visit/offset order.
The original sampler traverses consecutive pool rows, whereas the candidate
mixes cohorts within every batch. A measured difference cannot be attributed to
the marginal6000/2000 ratio alone. The original weak weights remain fixed; their
aggregate weighted contribution naturally changes with the allocation. The
result does not establish a generalization winner, estimate natural FAR or
supply independent human/SSC305 evidence.

## Shared execution and evidence

The existing augmentation engine accepts an explicit, immutable
`ExecutionPolicy` argument for its run, audit, queue and completed-evidence
functions. The default policy uses the original contract and entry point.
The allocation module supplies only its contract, recipe validator, schema,
entry point and source inventory. No global function is patched and no callable
is read from a JSON recipe.

The numerical training loop, forward/optimizer ledgers, per-draw loss checks,
final tensor observations and saved-output audit are reused. Their dependency
hashes must match the code retained by the actual control execution. The
parameterized engine is the one declared changed dependency; the original frozen
source directory and old execution artifacts are not modified or relabeled.
The new output retains both original-control source hashes and new execution
source hashes.

`compare` accepts completed, audited control and allocation evidence. It checks
the same row ordering, initial float-state identity, complete weak-label bindings
and features, then joins all402 saved conditions. It reports old/new train and
development separately; the three offsets remain repeated conditions. The
full paired ledger contains private row text and identities and must not be
copied into a public result document.

## Finite execution decision

The request has four fields:

```json
{
  "schema": "kws-c1-fixed-source-allocation-request-v1",
  "control": {"output": "/absolute/completed/control/output",
              "manifest_sha256": "<actual-control-manifest>",
              "audit": {"path": "/absolute/control-audit.json",
                        "bytes": 0, "sha256": "<actual-file-sha256>"}},
  "budget": {"path": "/absolute/stage-declaration.json",
             "bytes": 0, "sha256": "<actual-file-sha256>"},
  "decision": null
}
```

Pins use actual byte counts and SHA256 values; zeros above are placeholders.
`preview` accepts a null decision and writes a candidate with
`execution_authorized:false`. It cannot produce a runnable queue. `declare`
requires an actual pinned `kws-c1-allocation-decision-v1` file whose decision is
`run-single-fixed-allocation-3to1`, binding the saved posterior summary, the
same control evidence, stage budget and rationale. It also fixes
`one_model_invocation:true` and `no_threshold_or_loss_tuning:true`.
`do-not-run`, missing decisions and changed pins are rejected.
The posterior summary must be a completed `kws-c1-posterior-audit-v1` result
with zero new model constructions/forwards/optimizer steps, the exact original
paired-comparison and control-manifest bindings, and the full declared workload:
570 new saved numerical conditions,210 reused conditions and24 unscored unknown
conditions across804 paired model conditions. This validates completion and
inputs; the scientific rationale remains the recorded operator decision.

The current stage keeps CPython3.12.16, one heavy-model slot,4 CPU threads and1
interop thread. The candidate has one900-second queue attempt and an840-second
worker deadline; its audit is bounded at240 seconds. There are no new TTS/ASR
calls, model retries or ratio/loss/threshold sweeps. The stage's aggregate budget
includes separate CI jobs and is checked before launch by its owner. The module
verifies the pinned finite declaration and requires at least19 runner minutes
reserved for the single training attempt plus audit.

From the reviewed source and pinned runtime, metadata preparation uses:

```bash
python -I -S -B training/research_c1_source_allocation.py preview \
  --request /absolute/pending-request.json --output /absolute/candidate-recipe.json
python -I -S -B training/research_c1_source_allocation.py declare \
  --request /absolute/decided-request.json --output /absolute/frozen-recipe.json
python -I -S -B training/research_c1_source_allocation.py queue-request \
  --recipe /absolute/frozen-recipe.json --python /absolute/pinned/python \
  --output /absolute/queue-request.json
```

The owner records the actual posterior-informed decision and saved preparation
before initializing the unique queue. `run` retains the existing queue-only
identity and shared-lock checks. `audit` consumes a completed output with the
numeric runtime, while `compare` reads saved evidence without new inference.
No training result is implied by a successful preview or fixture.

The metadata fixture checks exact quotas, fair visits, deterministic offset
rotation, unchanged weights, unknown/development exclusion, source-family and
audio-identity leaks, fixed budgets, changed decision pins, pending previews,
the explicit queue entry point and retention of the original sampler.
