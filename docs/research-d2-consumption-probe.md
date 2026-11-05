# D2 verified full transcript consumption probe

The D2 source package has its own verifier and cannot enter the fixed 256-cell
D1 aggregator. `training/research_d2_inputs.py` replays every D2 package through
`active_d2_weak_view.verify_view`, then projects its observed complete transcript
with the unchanged `research_d1_vocab.fulltext_targets`. No model is run during
aggregation, preparation, declaration or source verification.

The fixed vocabulary is the original prepared 2601-output vocabulary, including
the predeclared 沃 and 灯 extension. Legacy 2599 consumer IDs and four-token
exports are never copied as targets. An ordinary negative contributes its whole
observed utterance to CTC. Unknown, failed-native, OOV or geometrically infeasible
rows remain in the all-row audit with `target_ids=null` and zero CTC and keyword
weights. Original observed labels, full text, masks, weights, and ASR evidence
bindings remain available under `observed_label` and the parent view.

Each package's producer source bytes must equal the verifier actually loaded.
Aggregation rejects repeated source IDs even across plans, repeated native PCM,
and cross-split family, speaker, reference, ancestor and PCM aliases. Original D1
source IDs and native PCM are included in this check when its prepared anchor
is supplied. Every row is replayed before filtering, including failed, uncertain
and OOV development rows. Batch-local `clip-000` IDs may repeat.

## Reuse the original initialization without changing D1 source pins

`research_d2_consume.py prepare` calls the unmodified D1 `verify_prepared` in an
isolated subprocess whose source root is the anchor's retained `execution-source`.
That original verifier checks its entire frozen manifest and all original
execution source hashes. Nothing monkeypatches its root, skips a source file or
weakens a historical assertion. The new D2 preparation copies only the authenticated
C1 initial checkpoint and original token bytes, and pins its own source, input,
projection, split, and initialization identities under a separate D2 schema.
The anchor and native weak packages remain read-only.

```sh
python3.12 -I -S -B training/research_d2_consume.py prepare \
  --anchor /private/original-d1-prepared \
  --anchor-manifest-sha256 ACTUAL_ORIGINAL_MANIFEST_SHA256 \
  --view-loader /checkout/kws-data/automation/active_d2_weak_view.py \
  --package /private/d2-chunk/weak-view \
  --package-manifest-sha256 ACTUAL_D2_VIEW_MANIFEST_SHA256 \
  --output /private/new-d2-prepared

python3.12 -I -S -B training/research_d2_consume.py declare \
  --prepared /private/new-d2-prepared \
  --prepared-manifest-sha256 ACTUAL_D2_PREPARATION_SHA256 \
  --output /private/new-d2-job.json
```

Repeated `--package` and `--package-manifest-sha256` pairs aggregate bounded D2
views. The consumption probe requires 1–64 eligible training rows. Development
rows are verified for isolation and retained but receive no model forward or
optimization in this probe. An ordinary development transcript with OOV remains
visible and cannot alter the vocabulary.

## Fixed actual consumption

Run the declared job only through the existing durable `research_queue.py` and
the shared heavy-model lock. Queue argv invokes `research_d2_consume.py run` with
`--recipe {recipe} --output {output_dir} --heavy-lock /private/shared.lock`.
Declare at most two runner minutes for this job. Its immutable settings are:

- Original C1 2601-output initialization, seed 20261105 and a fresh AdamW.
- Original logmel frontend and original full-text weighted CTC loop.
- Original learning rate 0.001, weight decay 0.01 and gradient clip 5.
- 25 optimizer steps, batch size 4, 100 cyclic train-only draws.
- A 120-second internal deadline, no continuation, sweep or automatic retry.

This is a data-consumption check, not a quality optimization or evidence of
generalization. Initial/final checkpoints, all projected labels, selected train
rows, per-draw full text and targets, masks/weights, loss contributions, gradient
norms, before/after saved feature/logit tensors and source bindings are retained.
Per-row target NLL, greedy outputs and blank occupancy are train-only diagnostics.

The queue declares `summary.json`, `consumption.jsonl`, `before.json`, `after.json`,
`final.pt` and `MANIFEST.json`. Failure preserves actual partial status/ledger and
does not manufacture success or missing draws.

```sh
python3.12 -I -B training/research_d2_consume.py audit \
  --output /private/d2-job/outputs \
  --manifest-sha256 ACTUAL_OUTPUT_MANIFEST_SHA256 \
  --receipt /private/new-readback.json
```

Readback verifies the unchanged sources and native weak views, full consumption
ledger, exact checkpoint deltas and saved-logit NLL/blank diagnostics. It creates
no model, forward, optimizer step, ASR request or TTS request. Unit tests use
fictional rows and explicit verifier doubles, not real-label certification.

## Observed first-chunk execution, 2026-11-05

The frozen probe ran once after the shared model slot became available. Queue
elapsed time was 10.693694550 seconds (0.1782282425 runner minutes), with exit 0,
no automatic retry and no budget overrun. The original learning rate, vocabulary,
initial checkpoint and all other declared settings stayed unchanged.

All eight native D2 rows were verified. Their original split was six training and
two development rows. Full-text projection admitted four training rows and one
development row; one development transcript contained an OOV character and two
other rows remained uncertain. All three ineligible rows retained their evidence
and had zero CTC and keyword weights. No development row received a forward pass.

The four training rows contributed 25 draws each: one full positive, one partial
negative and two ordinary negatives. Both ordinary negatives kept all eight
observed characters as CTC targets; the target lengths were 2, 4, 8 and 8.
The actual ledger contains 25 optimizer steps and 100 sample exposures. All 25
gradient norms were finite and nonzero (22.720314–96.526733), and all 16 trainable
parameter tensors changed. The initial/final parameter delta L2 was 9.193725.

This establishes actual data consumption, not recognition improvement. Mean
target CTC NLL changed from 734.715744 to 335.877914, while the final greedy
outputs were empty on all four training examples. Final exact sequence matches
were 0/4 and the blank argmax frame fraction was 1.0. There was no development
evaluation, ASR/TTS call, parameter sweep or additional optimization run.

A separate process verified the output manifest, original anchor and D2 source
packages again, replayed the entire consumption ledger, and recomputed the saved
logit diagnostics and checkpoint delta without constructing or forwarding a
model. It recorded zero new model forwards and zero new optimizer steps.
The 17 unit tests passed separately in normal, `-O` and `-OO` Python modes.

| Evidence | SHA-256 |
| --- | --- |
| Native weak-view manifest | `89d9422e869925ba0499e6767359686b7e18c92c87d7f4e199908555b8f594a7` |
| D2 preparation manifest | `76c31eba869a6f737bb628339c6dbed9fbfca5f396d1b726d937b1a6b1184cb2` |
| Frozen job recipe | `57bc03bb9852957392a49a3c1ce8cbbfa03b10726b31a8d3cb85c3da01c099b6` |
| Output manifest | `e04d8379390df4c1834b5b9efb3fb3445e041933b0a300387291783089be93c0` |
| Complete consumption ledger | `c289fc4a04fa59da2552cb6f68725fd4e60e8c768acbe2f77ebb62e59d062735` |
| Original initial checkpoint | `da7671fe71fef9ecd6822c747e3fe153b9ee563a6784c18eded90d3354486ecf` |
| Final checkpoint | `ae737d9227490ba8dd762c0631c3d093e746aa0d5be43b2ad1a47d65863efc17` |
| Full-text vocabulary | `104575dab3a9f7ed622d8fa29aa8a897e0590cca8aed9214b26ef4cd1c8b4e4b` |
| Independent readback receipt | `c0d3ffdff1c242448eb8e128979f31f85f5f2783d2bf0fbe7d6b5ec93bee91ce` |
| Anonymous delivery summary | `bd54765f4fdf496606fb08895d0e3e09d520afaa7f645de541b3739b13cff161` |

The private runtime bundle is named `consumption-probe-001`; its anonymous summary
contains no speaker identifiers, transcript text, audio paths or reference paths.
The implementation used by the job was commit
`791ff2c1635bcba6f30e58e6a6bace035a79bf3c`. Historical D1 code and results were
not edited by this work.
