# Independent review of numerical recovery v2

Date: 2026-10-07. This review concerns the new `k1-prefix-saved-evaluation-recovery-v2` source only. It does not execute saved rows or change the original failed attempt.

## Method verdict

The scoped adapter change is defensible for the explicitly declared rounded binary64 weight law. `softmax6` retains the original FP32 recovery, ordinary binary64 subtraction/exp/fsum/division, and all six columns. Only an actually produced zero activates an exact Fraction check of the recovered-FP32 logit difference <= -512. Positive results remain untouched, including synthetic positive tails below -512. No decision threshold, posterior floor, normalization choice, crop, terminal slot, reset selection, or semantic label changes were introduced.

The proof in `underflow-assessment.md` establishes the ideal discarded-tail bound 6*T*2^-512 in total variation, or twice that in L1, using e>2 and a maximum term of one. The implementation never uses this bound to alter the fixed gate. The bound does not certify other exp/subtraction/fsum/division rounding. The unchanged mathematical module still certifies only the exact normalized ratios of the six supplied binary64 weights. The recovery protocol states this distinction clearly.

The earlier reported attempt remains a numerical failure before any EOF, with partial row work and source exposure retained in the record. It is not transformed into a candidate pass or semantic failure. Recovery is cumulative attempt 2 with its own frozen source, release, and exclusive attempt ledger. The runner binds the prior ledger's declared hash; preservation of the actual prior files is additionally checked by the coordinator/review, not by reopening the original ledger inside every recovery execution.

## Scoped source inspection

The source diff adds exact checks only around zero exp/division outputs, counters split into exponential zeros and additional division-only zeros, counts of converted/affected rows, indexed conversion errors, per-observation failure details, cumulative-attempt metadata, and the prior-attempt hash. Callback state handling, fresh instances only at whole-observation boundaries, all-six normalization, every observation's fixed gate, source/hash/geometry checks, resource caps, and exclusive output behavior retain their original method.

The unchanged manifest fixes M1–M5,N1, all 537 rows, 57 callbacks, and six DP instances. The mathematical module is unchanged. No new inference, decoder, network call, subprocess operation, temporal selection, parameter search, or alternate decision path is present.

## Independently observed adapter checks

`underflow_mock_review.py` imports only the original and recovery runners. It executes no mathematical DP and opens no saved-logit source. Eight check groups passed:

- Four ordinary/positive-tail rows preserve exactly the original adapter's binary64 outputs
- A synthetic gap of 1000 produces a permitted exponential zero while retaining six columns
- A gap of 600 retains its positive term; crossing -512 never prunes a value
- A positive minimum-subnormal exponential that rounds to zero on division is accepted and counted separately
- A mocked zero at exactly -512 passes; the next representable FP32 value above -512 and a small gap reject
- Width, bool, nonfinite values, and FP32 overflow still reject
- Exponential/division-only/row counts are exact and conversion failures include observation/callback/row indices
- An exact rational finite-product example verifies the stated union/TV inequality algebra without a DP

The output `underflow-mock-results.json` records the tested runner and reviewer-test hashes. The original runner still hashes to `5125ed3f3237ee38611b628bc9f3d18dc34f50c036341381fd6991f310e8af8c`.

The final recovery suite was also observed: `python -B -m unittest -v test_runner.py` passed all 12 synthetic/adapter tests in 0.011 seconds. This includes one four-row synthetic test of the unchanged DP's tie/EOF/bound interface; the remaining cases exercise adapter behavior, a mocked verifier, temporary synthetic source/output files, and release guards. These are synthetic checks, not a saved-row replay. The suite is accurately labeled synthetic/adapter tests rather than pure mocks.

## Preservation checks

The original v1 runner, protocol, manifest, and tests were hashed and compared against the immutable first review record. All match. The unchanged mathematical module matches `e9edc93ffde5569cf7d931342b959738e3a40085957d3d6e1b275664d6729963`. The existing first-attempt ledger was read only for hashing and matches `cf9c92a0e8db14cd187d8bbbd606a4dc0739fc62781e0f5dae69404c0313dbc2`.

Source freezing and the complete recovery test result are recorded in `underflow-review-manifest.json`. This review does not execute or authorize recovery itself; the separately authorized coordinator owns its explicit release and single execution.
