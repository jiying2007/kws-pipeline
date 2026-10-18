# Controlled weak-data increment: no model promotion

This single bounded experiment used the previously retained CTC head as an immutable reference. Both new heads started from that checkpoint with fresh, identical optimizers. Each retained the original 146-record rehearsal schedule (880 presentations, 220 batch-of-four updates), plus 220 matched-weight auxiliary slots. The control reused fixed old examples; the treatment used eleven new machine-weak recordings from already-exposed training voice families. Eight were strict dual-ASR agreements, three were low-weight majority candidates, and one additional pronunciation-conflict recording remained unsupervised. Weights 0.5/0.1 are optimization weights, not probabilities of correctness.

The main loss, seed, optimizer steps, base schedule and fixed final checkpoint were unchanged. Global clipping at norm 5 coupled base and auxiliary gradients. Fixed competitor eligibility gave 420 control versus 360 treatment auxiliary CTC comparisons; this difference is disclosed rather than described as identical subcall counts.

## Results and decision

- On the same exposed 26-record development set, threshold 0.1 gave all three models 11/11 reference-target hits and three weak-reference negative events. Ten of the eleven targets were observed within the analytic body-time window; one required the synthetic suffix.
- At fixed threshold 0.7, all three gave 8/11 target hits, while weak-reference negative events were 0 for the retained model, 0 for control, and 1 for treatment.
- On the eleven new supervised training-source recordings, threshold 0.1 retained 5/5 targets and reduced near-reference events from 2 to 1 versus both the original model and control. Full-text exact counts were 5/11, 3/11 and 6/11 respectively. This is a fitting result, not independent generalization.
- A historical human-confirmed synthetic KW2 positive regressed from score 0.9663 to 0.3750 in control and 0.3385 in treatment; both lost it at threshold 0.7. Its node and decision structure changed, so the score change is not a same-alignment comparison. Both new heads also introduced a high-score trigger on a weak near-phrase reference.
- Ordinary-word diagnostics remained at zero events in both fixed profiles, while full-text exact counts rose from 93/796 to 107/796 and 109/796. These are exposed word diagnostics, not an ASR benchmark or continuous negative hours.
- The entire fixed 21-point development trade-off and all four analytic time windows are retained in the aggregate. Neither new model has a uniform advantage. No threshold was selected after this experiment.

Keep the original CTC model and the existing conservative OR research candidate. Neither new head is promoted. Improved CTC fit and a training near-phrase gain do not outweigh the protection regression and lack of development event gains. No additional seed, learning rate, epoch, threshold sweep or repeat was added.

## Retained research identities

- CTC model SHA-256: `a366d9f5ebe7027a496e3f119debcf1e19d2f7a79226c96b31a4ddb3a2757f77`
- OR supplementary KW2 head SHA-256: `12f97c535e122c0e097068243fdf49b5ebb9c7566ff7d9268129fee932334f74`
- Conservative OR: CTC 0.7 plus supplementary KW2 0.85; supplementary KW1 disabled. Same-keyword events within 500 ms retain the earliest, with CTC first at an identical time. Cross-keyword raw scores do not compete.
- Effect-priority reference: original CTC alone at 0.1.

The retained Torch research chain has separate continuous evidence recorded in this directory. This experiment did not run either new head on continuous recordings. It confers no new continuous validation on them.

## Limits

Reference labels retain their historical or machine-weak status; unknowns remain excluded from negative denominators. The 885-record mixture includes training, previously exposed development and diagnostics, so aggregate totals are not generalization accuracy. Full clips include a synthetic suffix; analytic required availability is distinct from measured real transport. Weak near-phrase events are not natural false alarms per hour. The two fixed profiles express research trade-offs, and zero weak negatives is not the sole product requirement.

No model here is a qualified production best. The strict C numerical chain and SSC305 CPU, memory and I/O validation remain open. The bounded experiment is complete; further same-family micro-tuning is not automatically justified. Public evidence contains aggregate results and hashes only, with no recordings, per-recording labels, weights, private paths or Library identifiers.
