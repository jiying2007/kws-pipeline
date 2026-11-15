# English63: one-sided background-gradient projection

Request **en63-projection48-20261115-v1** is a new, bounded mechanism experiment.
The completed request `en63-background48-20261115-v1` remains consumed and unchanged.
This request neither reruns nor extends it. Its four trained checkpoints are immutable
comparators, with original artifact, result-member, and checkpoint hashes in `protocol.json`.

## Question and fixed comparison

The previous weighted-background continuation reduced natural-speech proposals but
failed the fixed positive-preservation benefit gate. This experiment asks whether
removing the component of its background gradient that conflicts with the original
30-clip gradient changes that behavior. It does not select a model, threshold, data
partition, training duration, or gradient coefficient from the new result.

Train exactly two new arms, one GRU and one DS-C-TCN, for 48 optimizer updates each.
Both start again from the exact original online-pilot checkpoints, not a checkpoint
from the background study. Their seeds are unchanged: 20261114 and 20261215. Four
previously trained old-only and weighted-background checkpoints are copied as frozen
controls; they receive zero new optimizer updates.

Evaluate baseline, old-only control, weighted-background treatment, and projected
arm, in that order, for GRU and DS-C-TCN in each variant. The projected arm is tested
against the baseline and old-only control using the unchanged benefit conditions.
The previous weighted-background arm is a descriptive mechanism comparator only.

## Training inputs and arithmetic

The original 30 weak train-role TTS recordings are the only positive training input.
Their original three-class MIL cross entropy and train-only normalizer remain fixed.
The natural-background set is exactly the previous 128 three-second central crops,
from four records per each of 32 speakers, with the same lexical order, features,
32-example microbatches, weak background label, and coefficient 0.5.

For each update, let `g` be the gradient of the mean original-30 MIL cross entropy.
Set the parameter gradients to zero, then accumulate `h` from the four natural
microbatches with multiplier **0.125 per microbatch**, so that mathematically
`h = 0.5 * grad(mean background-128 MIL cross entropy)`. Compute global scalar
`d = dot(g,h)` and `n = dot(g,g)` in deterministic `model.named_parameters()` order.
The order must match the frozen exporter roster and is recorded with names, shapes,
offsets, and counts.

If `d < 0` and `n > 0`, use `h_prime = h - (d/n)*g`; otherwise use `h_prime = h`.
A zero reference norm leaves the background gradient unchanged. There is no epsilon,
small-gradient cutoff, per-tensor projection, extra loss weight, or parameter freeze.
All dot products use `math.fsum` of products of exact FP32 values converted to Python
doubles. The projection and `g+h_prime` addition use double precision; only the
combined gradient is rounded to FP32. Apply the original `clip_grad_norm_(1.0)` and
fresh AdamW with learning rate 0.001 and weight decay 0.01.

The original loss is computed with the same mean `F.cross_entropy` operation as the
prior study. A detached, unreduced cross entropy from the same logits provides a
per-class mean-loss diagnostic. It causes no additional neural forward or backward.
The recorded `old_loss + 0.5*background_loss` is a descriptive weighted loss value;
the projected vector is not claimed to be the gradient of that scalar objective.

The geometric condition applies to the combined gradient before clipping and
AdamW. AdamW's moments, per-coordinate scaling, and weight decay can change the
actual step's first-order relation to either loss. Therefore the experiment retains
actual parameter changes and reports their dot products with `g` and `h`; it does
not assume or enforce an old-loss guarantee after the optimizer.

## Saved vector evidence

`GRADIENTS.zip` has exactly `gru-gradients.npz` and `dsc_tcn-gradients.npz`.
Each is a safe numeric-only NPZ with six little-endian, C-order FP32 arrays shaped
`(48, parameter_count)`, with parameter counts 6531 and 4803:

- `old_gradient`
- `weighted_background_gradient`
- `combined_preclip`
- `actual_clipped_gradient`
- `actual_parameter_delta`
- `parameter_after`

The stdlib saved verifier recomputes all global projection scalars and the exact
FP32 combined vector. It checks the actual clipped vector and the actual FP32 clip
coefficient against the recorded Torch norm, with a scale-relative 1e-6 bound between
that norm and an independent double-precision norm. Projection residual and norm
checks also use fixed scale-relative 1e-6 bounds, not exact-zero residual demands.

The original checkpoint supplies the parameter state before the first update. Each
recorded delta must be exactly the FP32 difference between consecutive saved
parameter states; the last state must equal the final NPZ parameter payload byte
for byte. The verifier recomputes actual-step dot products and norms from these
vectors. Scalar diagnostics never affect execution, the fixed gate, or selection.

## Evaluation and fixed benefit gate

Every model pass uses the same 13 streams: one continuous natural-development
stream containing the same 102 clips from eight speakers, plus the same 12 isolated
TTS development clips, reset independently. The old C replay and decoder are
unchanged: 16 kHz PCM, 160-sample feed chunks, 200-frame window, at least 50 observed
frames, threshold 0.8, three confirmation frames, and 100 cooldown frames.

For each architecture separately, let `B`, `C`, and `P` be total natural proposals
for baseline, old-only control, and the projected arm. Research benefit requires:

- `B > 0`, `C > 0`, `2*P <= B`, and `2*P <= C`.
- For each keyword, projected natural proposals do not exceed either comparator.
- For each keyword, projected correct-trigger clip count is at least the larger
  comparator count and at least two of four clips.
- For each keyword, projected wrong-other-keyword clip count does not exceed either
  comparator count.
- Projected ordinary-background proposals do not exceed either comparator count.

The experiment-wide decision is `RESEARCH_BENEFIT` only if both architectures pass;
otherwise it is `NO_BENEFIT`. Either decision is a successfully completed experiment.
Counts are not substituted for one another, and the old weighted arm does not change
the comparator or positive-preservation requirements. Eight full traces are retained
as a closed `TRACES.zip` archive and decoded again by the saved verifier.

## Budget, execution, and scientific limits

Actual model/audio work runs only in the authorized public GitHub Actions consumer,
after a first-attempt consumption claim. The create-only producer accepts the **inner**
`background-continuation` directory as `--background-result-dir`. Local source checks
use only invented arithmetic, vectors, NPZs, and trace evidence; saved-result replay
is stdlib-only and performs zero model forwards and zero optimizer updates.

The fixed budget is 96 optimizer updates and 480 training forward calls, totaling
4,669,824 padded training frames. Each training worker has a hard 180-second wall
limit, including imports and evidence serialization; the two-worker group limit is
360 seconds. Native evaluation has exactly eight passes, 104 logical streams,
575,720 forward frames, 575,928 feed calls, and 92,147,872 supplied samples. Each
native pass is bounded by 120 seconds and the entire native group by 180 seconds.
The outer execution limit is 1200 seconds; saved verification is bounded by 180.
Completed checkpoints and tapes survive a later-arm failure. There is no automatic
retry, resume, result-driven extension, coefficient sweep, or stopping rule.

All 517 natural source clips were previously exposed. The speaker partition does
not make them an independent holdout. The TTS material remains weak-label research
material, with no new human gold or formal training admission. This experiment adds
no audio, TTS, ASR, IPA calls, independent English FRR, continuous negative FAR/hour
qualification, SSC305 board result, promotion, or runtime-default change.
