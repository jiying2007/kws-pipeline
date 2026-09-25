# Grounded negative-wake margin: bounded experiment

This experiment follows #302. It changes one training term, not the C decoder,
keyword thresholds, canonical trainer, dataset, training schedule or release gates.

## Predeclared treatment

Four serial cold-start trajectories on the frozen original train split:
`grounded-ctc`, all128 rows, RNN32x64/logmel, seed1337/2346, 600 epochs,
AdamW0.001/batch16, same random PCM lead/tail schedule and all six milestones.
Control: `--negative-margin-weight 0`. Treatment: weight0.1 with fixed confidence
ceiling0.50. These are training constants, not changes to runtime threshold0.55.

On each transcribed row, the added term penalizes a chronological acoustic path
for either full wake word UNLESS that word is the exact target. It operates only
inside the waveform-derived activity span. It does not relabel nonwake phonemes
as blanks. Existing activity CTC and non-speech blank supervision remain intact.
The max-path scorer is a SURROGATE; it omits runtime dominance/retention and is
not claimed to be exact C inference. Avoids the old positive startup-path reward.

Compare same initial state, batch/context schedule and source corpus. Apply the
existing #302 train/calibration-only epoch selector unchanged. Freeze each
selection receipt before evaluating test. No eligible checkpoint means no
candidate, not a forced last-epoch fallback. Preserve adverse results. A selected
candidate and a passing workflow do NOT confer release authority.

## Rejected decoder controls

Before defining the loss treatment, fixed #302 hosted selected models were tested
on TRAIN/CALIBRATION only. The blank-retention grid0.7/0.5/0.3 used identical C
posterior traces. At0.5 training recall fell from32/32 to7/32(seed1337) and18/32
(seed2346); at0.3 it fell to0/32 and3/32. Reject both.

Fuzzy-child costs -8.25/-12/-16 preserved most recall but increased seed2346
prior-context calibration FA from3 to4. Reject both alternatives.

Constructed pause guards concatenate same-voice, same-split transcripts `ni3 hao3`
+ `xiao3 wo1`, or `xiao3 wo1` + itself, preserving active audio with a40ms trim
guard and inserted0/80/200/400ms gaps. These are constructed full-word examples,
not natural phoneme-aligned recordings. Default model/decoder already miss many
400ms-pause examples. No test split or protected data was used in these probes.
Neither decoder setting was changed in production or adopted for the treatment.

## Executed local result: NOT adopted as the default recipe

Local Torch2.10.0+cpu. All four 600-epoch cold-start runs completed serially
on one machine using the frozen source tree
`65c1c1571cb0df49a267f397f1be1aa349b6e4de`. Code-file hashes for the trainer,
new loss and all numerical dependencies are retained in each checkpoint.
Publication adds the remote workflow wiring and this result section only;
those edits do not retrospectively change the recorded training source tree.

The existing #302 train/calibration-only selector chose epochs400/300 for
baseline/margin seed1337, and400/400 for seed2346. Decisions were written and
hashed before test evaluation, then checked unchanged afterward.

Fixed runtime0.55 threshold; entries are matched/expected; false accepts:

|Seed / split / context|Baseline|Margin0.1|
|---|---|---|
|1337 train, all silence leads and continuous|32/32;0|32/32;0|
|1337 calibration,1s or prior speech|14/16;0|14/16;0|
|1337 development test,1s or prior speech|15/16;0|15/16;0|
|2346 train, all silence leads and continuous|32/32;0|32/32;0|
|2346 calibration,1s|13/16;2|14/16;2|
|2346 calibration,prior speech|13/16;10|14/16;8|
|2346 development test,1s|15/16;5|13/16;4|
|2346 development test,prior speech|15/16;12|13/16;10|

Both selected margin models preserve original train-fit and startup-independence
checks. Seed1337 has NO event-quality improvement; an earlier selected epoch
is not a training-speed claim because the full trajectory still ran. Seed2346
reduces some false accepts but loses two test matches. This is not a Pareto
improvement, not a deployment fix, and not permission to adjust hyperparameters
using these test outcomes. The repeatedly used synthetic test remains development
feedback, never independent release qualification. Reused prior clips inflate
exposure/counts; these numbers are not FAR/hour or independent failure samples.

Four actual training times including milestone retention:95.67,101.49,96.64,
100.52 seconds. This excludes full C evaluation and is not a DUT benchmark.
Independent verification loaded28 checkpoints, validated byte-bound milestone
sets and recipes, re-scored288 retained C metric groups, and checked identical
per-seed initialization/corpus/shuffle/context schedule and sole treatment difference.

Constructed same-voice pause guards were applied after selection on train/cal only:
seed1337 baseline vs margin train80ms:16/16 vs15/16; train200ms:10/16 vs11/16;
calibration200ms:4/8 vs3/8. At400ms both are poor (train2/16 vs1/16;cal0/8).
Seed2346 pause outcomes are unchanged;400ms remains train6/16,cal1/8.
No decoder policy or selection rule was adjusted based on this additional test.
These artificial half-phrase combinations expose an unresolved limitation and
must not be presented as natural-speech pause qualification.

## Validation and remote protocol

New13 + milestone18 + startup21 + preceding14 + attribution16 + pool13 +
training-state18 unittest cases passed:113 total. Strict C ctest5/5 and the
existing supply-chain regression passed. A3-epoch actual before/after default
training comparison has identical floating tensors, epoch history, shuffle/context
hashes and KWM bytes. This is3-epoch no-op evidence, not a600-epoch or cross-vendor
repeatability promise. New loss is never executed when weight is zero.

The existing single-worker30-minute workflow keeps its original four trials and
adds only the two declared margin trials. Same input pool, same selector, decision
freeze-before-test, full selected-model context evaluation and final-epoch
learning checks remain required; no existing gate is relaxed or removed.
The audit workflow executes the new unit suite. Hosted Torch2.13.0 results must
be inspected separately; local and hosted measurements are not interchangeable.

No model is promoted, no canonical training default/decoder/shipping threshold
is changed, and #242 remains open. This opt-in experiment makes a concrete
hypothesis testable and preserves its negative result; it does not close the
remaining incomplete-phrase generalization or within-word pause problems.
