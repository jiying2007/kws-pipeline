# Local-reset diagnostic and counterfactual (2026-10-20)

## Outcome

The four-replay counterfactual **did not meet its preregistered screening
criterion**. One additional high-score event appeared, but the unchanged strict
source scorer did not count target recovery, and known regression was recorded.
The candidate is not promoted and the production decoder is unchanged.

## 747-row diagnostic

On the fixed 22.44-second synthetic-gap prefix, all six historical events were
reproduced exactly. At row 697, two windows formed the target K2 candidate with
score 0.3847483701042063, then reset local state before downstream rejection:
window 588 was rejected by threshold 0.7; window 630 was rejected by
deduplication. At row 698, the tail-token posterior reached 0.9996832609176636,
but three active windows reported no suffix. The maximum score above concerns
actually formed candidates in the fixed source-plus-tail observation window,
not suppressed paths or hypothetical alternatives. This observation alone did
not establish that retaining state would recover the target.

## Frozen four-replay comparison

Two controls and two candidate decoder replays used the existing saved
posteriors: 747 and 1,074 rows per implementation, 3,642 decoder rows total.
No new network, frontend, or softmax calls were made. Both controls reproduced
all saved event fields exactly. Threshold stayed 0.7; no finish or between-clip
reset was added.

The sole candidate change skips local_reset and last_active_pos assignment for
formed candidates scoring below 0.7. Downstream deduplication, threshold,
refractory handling, and acceptance remain unchanged; equality at 0.7 retains
the original behavior. This is not an unrestricted no-reset decoder.

- 747-row prefix: events increased from 6 to 7. The additional K2 event at row
  698 scored 0.9938043189157924. The original six events retain their scores,
  nodes, and clocks; only two window_start fields changed.
- 1,074-row prefix: all six existing events remained exact; no new event.
- Internal candidates increased from 18 to 257 and from 17 to 239 respectively;
  duplicate candidates increased from 7 to 244 and from 9 to 231. Stable output
  counts do not imply unchanged internal work or cost. These count ratios are
  not CPU slowdowns; native per-arm CPU timing was not measured.

The new event's last node is sample 335,040, exactly the original source's
exclusive end. The frozen scorer therefore marks it cross-seam and unmatched,
not a recovered target. **Unmatched does not mean a confirmed acoustic false
alarm.** Neither the boundary rule nor labels were changed after seeing results.

## Limits and next-decision boundary

These prefixes overlap exposed development material; they are not independent
validation. The 747-row prefix covers the source end plus 1,500 ms but truncates
the next source. The 1,074-row prefix ends only 300 ms after the target source
and cannot establish the entire 1,500 ms window. No natural-stream recall/FAR,
phase/history causal isolation, or natural-stream regression pass is established.

Any future timing-semantics change or added waiting mechanism requires separate
preregistration; neither is implemented or required to interpret this result.
This result does not justify retrospective relabeling or promotion. Independent saved-result integrity review
passed, while the screening outcome remains failed. This publication contains
only summaries and introduces no new experiment, audio, weights, or raw arrays.

Machine-readable summary: [summary.json](summary.json).
