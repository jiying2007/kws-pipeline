# B1074 paired prefix observation (2026-10-20)

## Result

The frozen network logits/cache numerical gate **fails**. Separately, all six
emitted events are stable on this one exposed 32.24-second development prefix
at threshold 0.7. Event stability does not override the numerical failure.

- Logits: 1,715 / 2,791,326 values exceed tolerance, across 165 / 1,074 rows.
- Cache: 40,213 / 6,048,768 values exceed tolerance, across 1,073 / 1,074 rows.
- Unchanged gate: `abs(C - Torch) <= 1e-5 + 1e-5 * abs(Torch)`.
- Maximum absolute errors: logits 0.00030517578125; cache 0.00009918212890625.
- All six paired events match in keyword, nodes, and clocks. Historical C
  events reproduce exactly, including scores. Maximum paired event-score
  difference is 0.0000006098042450997099.
- Maximum posterior absolute difference is 0.000004231929779052734;
  top-3 changes, target floor crossings, and target top-3 membership changes are zero.

## What was compared

Model arm B, 1,074 network steps per side, 515,840 samples at 16 kHz, fixed
320-sample feeds, original synthetic gaps, no between-clip reset and no finish.
C and Torch network implementations use the same saved context and shared C
frontend, softmax, and decoder. The Torch frontend was not tested.
The publication base is
`a8c68c93de2c9218dd7e696bf3643d61e1a04d1c`. Execution dependencies were
frozen separately; this commit does not identify the complete runtime source
provenance.

Two historical, known-context, single-user whole-phrase confirmations remain
supported by matching events. They share a synthetic reference voice and add
zero independent positive examples. One event is emitted within its original
clip; the other is emitted in the following synthetic gap. The other ten
labels are not promoted.

An independent saved-result integrity review passed without running the
model, frontend, or decoder, or importing Torch. Its numerical recomputation
confirmed failure, not a numerical pass. Public preparation ran no scientific
experiment. This directory contains only a summary, not raw reproducibility
artifacts.

## Limits

This is not independent acceptance, natural-stream recall/FAR evidence,
whole-model/all-layer numerical validation, real-number accuracy, or product
qualification. Prior B76 failure and the rejected FP64 candidate remain
unchanged. No threshold relaxation, tail extension, or rerun is implied.

The recorded CPU 17.076947 seconds and wall 15.775962 seconds are late-stage
snapshots: CPU is post-comparison; wall precedes final receipt writes and
checks. Neither is an exact final total or a performance benchmark. Private audio,
weights, arrays, filesystem paths, and delivery identifiers are excluded.

Machine-readable details: [summary.json](summary.json).
