# Bounded local retention: research-only implementation

This package retains an opt-in research patch and synthetic native tests for a
fixed five-posterior-row low-score retention policy. It does not change the
production decoder or model defaults. **The saved development screen remains
failed; the candidate is not promoted.**

## Reproduce source and synthetic state checks

From the repository root, run:

```
python3 -B research/bounded-local-retention-2026-10-20/run_synthetic.py --cc gcc
python3 -B research/bounded-local-retention-2026-10-20/run_synthetic.py --cc clang
```

The driver reconstructs the research candidate from the existing, hash-pinned
public C source plus this directory's patch and headers in an ephemeral build
directory. It checks source identity and runs only invented-state native cases.
It does not load models, private posteriors, audio, or scientific replay inputs.
The existing research CI compiler matrix invokes the same checks. The patch
retains diagnostic instrumentation so that its reconstructed candidate matches
the implementation used in the saved observation; the instrumentation is not a
claim about production footprint.

## Fixed policy and scope

A legal low-score candidate opens a nonrenewable five-row episode per physical
window. Repeated low scores do not extend its deadline. A strong candidate keeps
the original strong-reset behavior. At the deadline, the formed candidate is
copied, then retained local state expires after detect and before global merge;
the copied candidate still reaches the unchanged merge logic. An overdue or invalid
clock is rejected rather than rescued by a late candidate. Window retirement
right-censors live episodes without extending window lifetime. Clocks and
counters commit transactionally with the original row state.

The threshold remains 0.7. Beam search, suffix detection, merge, deduplication,
and downstream threshold/refractory behavior are retained. No new finish,
flush, padding, or network operation is introduced. Five rows correspond to
150 ms nominal audio-coordinate progression, not device wall-clock latency.

## Saved observation, independently reviewed

Two candidate-only decoder replays used already saved 747-row and 1,074-row
posteriors, 1,821 rows total; previously verified controls were reused. There
were no new network, frontend, or softmax calls and no new control replay.

- 747-row prefix: all 16 fields of each original six events remain exact. One
  additional K2 event appears at row 698, score 0.9938043189157924.
- 1,074-row prefix: all six original events remain exact; no additional event.
- Internal candidate counts, original / without deadline / five-row deadline:
  18 / 257 / 40 and 17 / 239 / 32.
- Duplicate candidate counts in the same order: 7 / 244 / 27 and 9 / 231 / 24.
- Six retention episodes occur on the 747-row prefix: four expire and two close
  on strong candidates. All three episodes on the 1,074-row prefix expire.

The new event's last node is sample 335,040, exactly the source's exclusive end.
The frozen strict scorer still marks it unmatched, not recovered. This does
not establish a confirmed acoustic false alarm. No label or time boundary was
changed after observing the result.

## Evidence limits

These overlapping exposed prefixes are development evidence, not independent
validation or natural-stream recall/FAR measurement. Candidate-count reductions
are not measured CPU speedups; no native per-arm CPU ratio was established.
The five-row bound does not certify a 500 ms product wall-time requirement.
Source reconstruction and synthetic correctness checks cannot override the
failed scientific screen or authorize a production default change.

Public aggregate: [summary.json](summary.json). Source license and attribution
remain those of the pinned upstream research source; see its LICENSE and NOTICE.
