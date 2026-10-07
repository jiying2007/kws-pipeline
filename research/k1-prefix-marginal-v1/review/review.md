# Independent mathematical and implementation review

Scope: synthetic-only review of the first-prefix CTC research verifier on 2026-10-07, plus source/method review of the proposed saved runner and its structural manifest. No saved posterior values, audio, model or decoder run, repository integration, or public write was used.

## Result

The ten-state mathematical recurrence is correct for the declared six-token CTC path law and A/B/R first-prefix classes. The revised implementation passes the independent raw-path oracle and focused numeric/lifecycle checks below. An exact tie defect was found in the initial floating-log decision, then removed by rigorous integer interval certificates. A boundary that the finite certificate cannot resolve is explicitly `NUMERICALLY_UNRESOLVED` and fails closed; it is not represented as a certified mathematical negative.

## Mathematical check

Each raw path emits a character only when the current symbol is nonblank and differs from the immediately preceding raw symbol. Blank therefore resets repeat handling. Before absorption, each nonempty valid prefix can end only in its final label or blank. This gives the seven transient states e, b1/r1, b2/r2, b3/r3 and the three absorbing A, B, D states. The direct nonnegative D flux correctly counts re-emission after a blank and excludes an adjacent raw repeat. Every outgoing column sums to one after all six weights are normalized.

A and B are exactly the collapsed-string starts-with events `你好小窝` and `你好小屋`. R contains D plus every incomplete prefix, so the partition is exhaustive and disjoint. A three-class plurality is never substituted: the decision is the binary comparison A > B+R, equivalently A > 1/2 in the exact partition. The fixture with (A,B,R)=(.4,.3,.3) rejects.

The first-prefix classes absorb every suffix. Therefore A and B can only increase on appended fixed posterior rows, and R can only decrease. In particular, the synthetic (.6,0,.4) then (.6,.4,0) construction remains an accept despite the later certain 屋. Identical posterior inputs cannot identify whether a later label belongs to the same acoustic occurrence or a separate word. These limitations are mathematical properties, not implementation defects or passing confuser-protection results. No M3 repair, acoustic boundary, same-occurrence inference, continuous detection qualification, calibration, latency, or real-data accuracy follows.

## Independent oracle and observations

`independent_review.py` enumerates every one of the 6^T raw paths, including zero-weight paths. It computes an exact integer path numerator over a common rational denominator, collapses adjacent identical raw symbols before removing blank, and uses ordinary string `startswith` for the buckets. It does not call the production DFA or recurrence. A separately transcribed Fraction recurrence checks the design and supplies per-frame exact states for interval enclosure tests.

The bounded suite has 585 exhaustively generated matrices: deterministic 你 好 followed by zero through three rows selected from eight fixed rows (the six deterministic rows, an all-six uniform row, and a blank/窝/屋 mixture). This covers a finite menu, not all possible posterior matrices. Another 27 directed cases cover empty input, leading blank, repeats and blank-separated repeats, alignment marginalization, exact ties, all-six nonuniform mixtures, suffix absorption, and complete 4–7-row mixtures.

Observed main-suite result: 612 cases, 5,523,157 raw paths explicitly visited, 619,348 nonzero paths, no remaining failures. Every exact rational state was enclosed at every tested frame. Every two-way chunk split, including empty chunks, reproduced bit-identical log states and integer bounds. EOF decisions were 20 certified accepts, 586 certified rejects, and six explicitly unresolved exact ties. The eight mathematical tie cases all failed closed; two were exactly certified negative and six were unresolved at the fixed precision.

The original implementation accepted this exact tie:

1. Deterministic 你 好 小
2. Literal weights 窝=1, blank=4
3. Literal weights 窝=3, blank=5

The exact A mass is 1-(4/5)(5/8)=1/2, B=0, R=1/2. Floating readouts were A=.5000000000000001 and R=.49999999999999994. The initial strict log comparison accepted. The revised certificate returns `REJECT_K1` with `NUMERICALLY_UNRESOLVED`, so rounding cannot turn this tie into an acceptance.

`edge_review.py` additionally checked:

- Missing/reordered/unknown token maps and 15 invalid row forms, including wrong widths, negative, NaN, infinities, bool, string, complex, overflowing and underflowing conversions, dictionary, and missing input
- Atomic rollback for bad single rows, a bad row after a valid chunk row, and row/chunk iterators that raise
- Empty chunks, finalization with no synthetic transition, only one emission, repeated EOF, updates blocked after EOF, and reset from partial or finalized observations
- The exact 4096-row limit and rollback on an overflowing chunk
- A nine-row deterministic first confuser followed by a wake, which remains entirely B
- Seven-value bounded reads for malformed unbounded row iterables
- Four extra short exact oracles, 5,184 raw paths, for maximum and smallest-subnormal float weights, including a finite-sum overflow situation and a tiny A whose rendered value is zero but log_A is approximately -5816.891139259061
- A=1/2 and the nearest representable binary64 terminal weights on either side; their decisions are reject, reject, accept
- Exact power-of-two per-row scaling across a broad exponent range
- 4096 nontrivial all-six rows with conservation and monotone absorbed masses

The validation tolerance for approximate readouts is 2e-12 in the short suite; the long conservation check uses 1e-10. These tolerances are assertions about floating arithmetic only. Neither is used for the decision.

## Numeric certificate proof and limits

The production module retains ten approximate log accumulators and adds ten lower plus ten upper integer bounds at scale S=2^256. This is 30 numeric state values plus lifecycle metadata; there is no posterior history. The model still has ten states. Literal Real inputs first convert to binary64 under an explicit contract; the certificate applies to those resulting binary64 values.

Each finite binary64 weight has an exact integer numerator and power-of-two denominator. Taking a common denominator across all six values creates exact integer relative weights. The certificate normalizes their full sum and applies the nonnegative transition using exact Python integer operations, flooring lower outputs and ceiling upper outputs. By induction, every true state mass is enclosed. Integer arithmetic does not rely on libm accuracy or the floating log path.

The exact transition is column-stochastic. If the previous one-sided L1 bound error is E, the next error is at most E+10/S: each of ten outputs introduces less than one grid unit of rounding. Therefore each one-sided L1 error is at most 10T/S, and the total enclosure width is at most 20T/S. A statement that the full width is 10T/S would not follow from this proof. At T≤4096 these bounds are finite and each stored bound numerator stays at most S+40960, requiring at most 257 bits. Temporary normalization integers are also bounded by the binary64 exponent range.

At EOF the code accepts only if lower_A > S/2, certifies rejection if upper_A <= S/2, and otherwise returns an explicitly unresolved fail-closed result. Thus exact mathematical ties cannot accept. A sufficiently close genuine positive can be unresolved; this is an honest arithmetic limitation, not a new learned threshold. The certificate precision is fixed before data and never fitted to examples.

Log masses remain approximate diagnostics, including when their exponentials underflow. The EOF decision never reads the approximate logs. No softmax wrapper is part of the module; a separate conversion from logits would not gain a mathematical certificate for its pre-conversion real values merely by feeding the resulting floats here. Input semantics, token mapping, and whole-observation boundaries still require external provenance.

## Review boundary

This review supports the synthetic implementation gate only. It does not authorize or report saved-data evaluation, new inference, model/decoder changes, training, integration, deployment, or public writes. Any later saved-data stage must use its separately authorized fixed whole observations and declared semantics without crop, reset, threshold, class-grouping, or normalization search.

## Saved runner source and method review

At the parent's request, `k1-prefix-saved-evaluation-v1/runner.py`, `protocol.md`, `manifest.json`, and `test_runner.py` were reviewed without reading either raw saved source. The manifest describes six complete observations in the fixed M1–M5,N1 order, 537 rows in 57 callbacks. Source provenance in this review is limited to those prepared metadata and declared hashes; the raw collector/model sources were not independently reopened here.

No blocking method or source defect was found. The runner consumes every pinned callback's complete rows in file order, carries state across callbacks, and creates one fresh verifier per whole observation. It recovers serialized FP32 values and applies stable six-way binary64 softmax at unit temperature. Underflow, overflow, nonfinite input, geometry mismatch, source hash mismatch, and unresolved decisions fail rather than triggering a normalization or window search. The binary64-softmax certificate limitation is stated explicitly.

Every manifest label must get its corresponding certified binary action for the gate to pass; M3/N1 additionally have a reported primary gate. All six observations are reported once, including failed labels. The code does not crop, search thresholds, tune resets, choose terminal slots, evaluate a three-class alternative, call a decoder/model, launch subprocesses, or access the network.

The explicit release pins manifest, protocol, runner, tests, and module hashes. The attempt ledger is opened exclusively before raw input reads; result creation is also exclusive. The runner bounds input bytes and output size, and sets 5 CPU seconds, 15 wall seconds, 64 MiB address space, and 1 MiB output. A hard process kill may leave only a partial ledger; the protocol correctly treats that as a consumed failed attempt, with no retry. Release, ledger, and actual evaluation remain owned by the coordinator.

Observed: `python -B -m unittest -v test_runner.py` passed seven pure mock tests in 0.005 seconds. They covered disarmed no-read behavior, FP32/full-six softmax, exclusive output files, mock source geometry/hash validation, unresolved gate failure, unarmed release refusal, and hash rejection for every reviewed source/module file. No actual saved values were opened or evaluated by this reviewer.

Final source and review hashes, plus the exact commands observed, are recorded in `review-manifest.json`.
