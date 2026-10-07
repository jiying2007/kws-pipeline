# Guarded binary64 underflow repair assessment

Prepared after the reported first saved attempt stopped at a softmax-underflow guard. This assessment does not revise, delete, rerun, or reclassify that failed attempt. The original release, runner, protocol, source pins, and ledger remain unchanged. No saved value file, real-data DP, model, or decoder was run for this assessment.

## Recommendation

Use a separately reviewed recovery adapter with the original arithmetic order: recover each stored FP32 value, promote to binary64, subtract the row maximum, apply `math.exp`, sum all six with `math.fsum`, and divide all six terms by that sum. Permit zeros that these operations actually produce, under the narrow validation condition below. Keep the mathematical module unchanged and certify only the exact normalized ratios of the resulting six binary64 weights.

For every coordinate whose computed exponential or final divided weight is zero, compare the exact recovered-FP32 values with rational arithmetic. Require x_i - max(x) <= -512. Otherwise stop. Do not zero any positive value merely because its logit difference is below -512. The condition is a sanity check on an already produced arithmetic zero, not a pruning rule, probability floor, fitted model threshold, or alternate normalization. The fixed A > 1/2 decision, complete rows, observation boundaries, labels, gate, and unresolved handling remain unchanged.

This is smaller and easier to audit than introducing a second log-input DP and a full transcendental interval implementation. It explicitly defines a rounded-weight model law. The earlier protocol's fatal-zero guard changes, so the recovery requires a new frozen protocol/release and leaves the original numerical failure visible.

## Rigorous discarded-tail bound

Let exact recovered logits be x_i, m=max_i x_i, ideal unnormalized terms u_i=exp(x_i-m), and ideal full softmax p_i=u_i/sum_j u_j. Let Z contain only coordinates for which this adapter actually produced zero and validated the exact difference condition.

For i in Z, x_i-m <= -512. Since e>2,

    u_i <= e^-512 < 2^-512.

At least one exact term is 1, so the ideal denominator is at least 1. Consequently the discarded ideal probability mass is

    d = sum_{i in Z} p_i < |Z| * 2^-512 <= 6 * 2^-512.

If q is the ideal distribution formed by dropping those same coordinates from p and renormalizing the remaining exact terms, then

    TV(p,q) = d,
    ||p-q||_1 = 2d.

For T independent frame rows, the product path laws differ in TV by at most the sum of their frame TV distances, hence at most 6*T*2^-512. Applying deterministic CTC collapse and A/B/R bucketing cannot increase TV. Thus each bucket's absolute change from this ideal tail removal is at most 6*T*2^-512; the bucket-vector L1 bound is twice that.

These are ideal discarded-tail comparisons, not an error certificate for the entire computed adapter. They do not bound the other effects of binary64 subtraction, `math.exp`, `math.fsum`, or division on retained positive values. The actual verifier's interval certificate remains exclusively about its supplied rounded binary64 weight law. No claim of exact real-softmax classification should be made.

The per-row TV tail bound is about 4.4750e-154. For the longest predeclared observation, T=109, it is about 4.8778e-152. The generic module limit T=4096 gives about 1.8330e-150. Summing all 537 rows gives about 2.4031e-151 as a cross-observation bookkeeping upper bound, not one shared-observation DP. These are much smaller than the existing fixed-point interval scale, but that comparison does not certify unrelated ordinary floating rounding.

## Why not claim half a minimum subnormal

A bound of one half of the minimum binary64 subnormal per dropped exponential would require a correctly rounded exponential guarantee. Python's `math` API delegates largely to platform C math functions and does not supply that guarantee. The -512 proof avoids such an assumption. Python's `Decimal.exp` does document correct rounding, but using it to certify all transcendental and normalization steps would materially increase the implementation and review burden for this narrow repair.

Sources: [Python math documentation](https://docs.python.org/3/library/math.html#module-math) and [Decimal.exp documentation](https://docs.python.org/3/library/decimal.html#decimal.Decimal.exp). The tail/TV proof and repair design above are derived here.

## Static accounting for the reported failed attempt

The parent reported the first failure at zero-based M1 callback 6, row 9. The frozen manifest has nine rows in callback 0 and ten in callbacks 1–6. Therefore callbacks 0–5 committed 59 rows; callback 6 made nine provisional successful DP transitions before its final row failed during softmax. The failed coordinate belonged to the 69th attempted softmax row. In total, 68 row transitions were computed, 59 committed, and no EOF result was reached. Chunk atomicity discarded the nine provisional transitions.

This does not mean that only 69 rows were loaded or exposed to the process. The runner reads, hashes, and parses both complete saved files before numerical evaluation, so all 537 saved rows had already been loaded. These counts are derived from source control flow, structural counts, and the parent's reported failure location; this reviewer did not reopen the actual data or rerun it.

## Required focused mock verification

- Ordinary finite rows produce exactly the original adapter's binary64 weights
- A synthetic gap of 1000 permits an exponential zero
- A gap of 600 retains its positive value even though it is below the sanity guard
- A smallest positive exponential that becomes zero on division is permitted after the same exact-difference check
- A mocked zero at a small gap is rejected; guard behavior at exactly -512 and the next representable FP32 value above it is checked
- All six positions remain present; finite input, width, bool, overflow, and nonfinite validation remain strict
- No actual mathematical module, saved-data replay, model, or decoder is run in these adapter mocks

This document is a method recommendation. Recovery source review, hashes, and observed mock results must be recorded separately before the coordinator considers a newly authorized recovery.
