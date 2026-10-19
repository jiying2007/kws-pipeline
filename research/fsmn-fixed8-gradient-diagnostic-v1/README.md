# Fixed8 gradient-only diagnosis, 2026-10-19

This experiment is separate from the continuous-stream experiment. Three frozen head states (original B, historical 37-step control, historical 37-step KL treatment) were measured on the same eight deliberately balanced training-only batches. No optimizer, parameter update, encoder forward pass or protection evaluation was performed.

## Findings

| Frozen state | Mean weighted KL/base gradient norm (%) | Batch range (%) | Minimum base/total cosine | Hypothetical clip5 batches |
|---|---:|---:|---:|---:|
| Historical control | 0.3661 | 0.2634–0.5286 | 0.9999861 | 8/8 |
| Historical KL treatment | 0.2316 | 0.1598–0.3091 | 0.9999952 | 8/8 |

The ratio is 100 × norm(0.1 × KL gradient) / norm(base gradient), with base = CTC + 0.1 × competition. The mean is the mean of eight batch ratios, not the ratio of mean norms. KL/base cosines span −0.1808 to 0.0428 for control and −0.2733 to 0.0652 for treatment: the regularizer is not always aligned with the base gradient. Its addition only slightly rotates the total direction at these measured endpoints.

B is its own teacher: KL loss is exactly zero while float32 differentiation leaves residual KL gradient norms of 1.41e−7 to 2.55e−7. B's KL cosines are omitted from the anonymous results because their directions have no substantive interpretation. Cosines numerically within roundoff of one should not be overinterpreted.

All 24 measured total gradients would exceed norm 5. The reported clipping coefficient and effective norm are hypothetical scalar clipping calculations. Uniform global clipping scales all components together; it does not itself erase their relative norm ratio or direction. No Adam state or update was measured. Actual parameter updates were zero and all three heads' initial/final hashes matched.

## Design, accounting and limits

Each batch contains two targets, one historical competitor, one confusable and four ordinary clips. There are 64 presentations of 54 unique clips per state, from the historical 146-clip training roster. Three samples per batch are competition-eligible; all sample weights are one. Labels remain historical/weak. Repeated clips and the deliberately balanced sampling are not the historical 37-step empirical distribution, independent test data or evidence of acoustic label truth.

CTC is target-token normalized then averaged over all eight samples. Competition is also averaged over all eight, not only the three eligible samples. KL is the equal mixture of conditional-nonblank and binary blank/nonblank teacher-to-student KL at temperature one; valid frames are averaged per sample, then samples are averaged.

The single completed execution recorded 4 loads, 8 teacher-head forwards, 24 student-head forwards, 24 full CTC calls, 138 competitor CTC calls and 72 gradient traversals. All started/completed counts agree; 565 ordered checkpoints contain 24 measurements. Host wall time was 7.582 seconds and maximum sampled process-group RSS was 432,959,488 bytes at a 0.2-second sampling interval, not an instantaneous hard memory bound.

This is a post-evidence endpoint diagnosis. Smaller KL gradients in the historical treatment are compatible with being closer to its teacher under this metric; they do not establish a causal benefit, absence of benefit, useful larger lambda or a historical training-trajectory explanation. No protection-set-based lambda tuning is justified.

A useful next experiment would predeclare a matched balanced schedule comparing lambda 0 and the unchanged lambda 0.1, fixing all other training choices and evaluation gates before execution. It would test sampling under controlled conditions rather than choose a larger lambda from these endpoint ratios. If resource limits permit only one arm, a balanced lambda-0 control can test that sampling change, but cannot isolate the KL effect. Any next numerical experiment requires its own review and release.

## Publication boundary

`results.json` contains anonymous per-batch scalar measurements and aggregates only. No recording IDs, transcripts, source paths, source PCM digests, cache tensors or weights are included. Saved-result review checked ledger completion, unchanged head identities, source hashes, scalar norm expansions, dot/cosine relationships, clipping and aggregates without rerunning tensors. Private reproduction evidence is archived separately. These two files are intended for their own research-evidence change.
