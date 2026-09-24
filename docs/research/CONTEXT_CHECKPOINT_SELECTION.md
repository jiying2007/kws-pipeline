# Context checkpoint selection (development only)

The startup-context trainer previously overwrote each rolling 100-epoch snapshot.
The new opt-in `--retain-milestones` preserves epochs 100, 200, 300, 400, 500 and
600 for a 600-epoch trajectory (at most eight for the bounded trainer). It keeps
checkpoint, canonical exported KWM, provenance and a complete byte-bound index.
Incomplete indices, missing epochs or changed tensors/exports fail closed.
This is diagnostic retention, NOT optimizer/RNG-continuous resume.

## Predeclared decision rule

`training/select_context_checkpoint.py` evaluates only a filtered train/calibration
view. It never evaluates test rows. At unchanged 0.55 keyword-pack thresholds it
uses three identical arms for every epoch: 1-second silence, a lexically safe
same-split nonwake predecessor, and the existing continuous mixed stream with
2-second gaps. The canonical C event matcher supplies all metrics; each arm's
references and detections are retained.

Eligibility requires complete train-fit with zero false accepts in all three
arms, and at least 75% observed recall for EACH calibration keyword in EACH arm.
The floor is an exploratory selection constraint, not product qualification.
Rank eligible epochs lexicographically by worst calibration FA count, total
calibration FA count, worst per-keyword FRR, total false rejects, then earlier
epoch. The loss and final-epoch convention do not participate. If none is
eligible, `selected=null`; a silent model is never a fallback winner.

All candidates must come from one byte-verified training trajectory and frozen
pool. Selection is made separately for the two predeclared seeds 1337/2346; there
is no search for a lucky seed. No threshold or decoder parameters are tuned.

The decision is persisted and hashed BEFORE evaluating the selected checkpoint
on test data. Existing startup, continuous, silence and counterfactual regressions
are rerun on the selected model; its actual train-learnability gate must pass.
The final epoch remains a separately evaluated reference. A selection failing a
post-selection check is a failed experiment, not silently replaced using test
results. Existing final-epoch gates are retained, not weakened.

## Limits

The frozen synthetic calibration/test data have already been used as development
feedback; neither is a fresh independent release holdout. Repeated predecessors
inflate FA counts/exposure and are NOT independent hard negatives or product FAR.
This study can reduce bad epoch selection; it cannot establish broad acoustic
robustness or fix decoder boundary handling by itself. No oracle state reset,
shipping-model promotion or cross-vendor reproducibility claim is introduced.
