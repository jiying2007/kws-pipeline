# K1 fixed-slot CTC evidence scorer, research v1

Standalone C11 math only: `k1_fixed_slot_ctc.h`, `k1_fixed_slot_ctc.c`, and
`test_k1_fixed_slot_ctc.c`, with an independent oracle in `review/`.
Dependencies are limited to the C library/libm. The scorer performs no allocation
or I/O. This source distribution contains deterministic synthetic tests; it
contains no model/decoder replay, real or saved data, parameter/window search,
or deployment integration. K2 is outside this module.

`RECEIPT.txt` preserves the original implementation-freeze build receipt
verbatim, including its pre-publication README hash and execution scope.
`review/REVIEW.md` preserves the independent mathematical review with public
relative test paths and generic scope wording. The implementation, header, both
oracle source files, and receipt are byte-identical to the reviewed versions.
This README's packaging clarifications do not change the mathematical contract,
numeric limits, or status meanings.

## Exact chosen contract

The caller supplies a single immutable raw CTC prefix path W occupying absolute
rows `[prefix_begin,a)`, its preceding raw label, and a sealed slot `[a,b)`.
All coordinates are unsigned 64-bit half-open row coordinates. The scorer checks
that W's **additional** CTC collapse, starting from the declared preceding raw
label, is exactly `你好小`; adjacent repeats merge before blanks are removed.
Every chosen W label must have strictly positive support. W's final raw label
must be `小` or blank. Its run is declared closed at `a` by the caller.

For each q in `{窝,屋}`, the terminal raw-path language is exactly
`blank* q+ blank*` on the **same** `[a,b)`. There is one nonempty contiguous q run.
There is no continued `小` inside the terminal slot, even if W ends in raw `小`.
There is no q-blank-q path, other label, or later-label veto in this language.
These exclusions define this research specialization; they are not a general
search over every CTC alignment of the full phrase.

Probability-domain recurrence, with old values on every right-hand side:

```
(pre, run, post) = (1, 0, 0)
pre'  = p(blank) * pre
run'  = p(q) * (pre + run)
post' = p(blank) * (run + post)
Zq = run + post
```

The implementation computes that recurrence in binary64 log space with stable
log-add. Inputs are binary32 probabilities in the fixed order
`[blank,你,好,小,窝,屋]`, plus caller-provided availability ticks. Every consumed
row's six values must be finite, in `[0,1]`, and sum in binary64 to within `1e-5`
of one. Nothing is silently normalized. Probability on other labels is excluded
from the chosen path language without redistribution to blank/窝/屋. Zero remains
zero (`-INFINITY` in logs); there is no probability floor. The accepted FP32 sum
tolerance means inputs can be slightly nonunit and returned quantities are the
literal supplied path masses, with that tolerance, rather than renormalized
probabilities. An allowed tiny row over-sum can produce a tiny positive log
mass; it is not clamped. No ratio or calibrated confidence is returned.
Build without fast-math, with IEEE NaN/infinity semantics and
preserved subnormal inputs; the verified builds use the default rounding mode.

## Returned quantities and statuses

All logarithms are natural. Branch index 0 means 窝; index 1 means 屋.

- `log_prefix_mass = log P(W)`, the product for that **one** supplied path,
  conditional on the supplied posterior rows and entry raw state. It is not
  the summed probability of all prefixes collapsing to `你好小`
- `log_conditional_mass[q] = log Zq`, the terminal family mass conditional on
  this exact W under the CTC path-product model; it is **not** normalized over
  the two branch families or over the three terminal symbols
- `log_joint_mass[q] = log P(W) + log Zq`
- Support is exactly `EMPTY_INTERVAL`, `NEITHER`, `WO_ONLY`, `WU_ONLY`, or `BOTH`
  and is determined from mathematical zero versus positive support, even when
  the product would underflow binary64. A tie returns two equal masses with
  `BOTH`, without selecting a winner

An empty slot succeeds with `EMPTY_INTERVAL` and both terminal/joint logs at
negative infinity. A nonempty all-blank slot succeeds with `NEITHER` and the
same zero masses. One zero branch stays at negative infinity. An unsupported W
is `K1_ERR_PREFIX_UNSUPPORTED`, so conditioning on a zero-support W never occurs.
On every error, the output bytes are unchanged. Successful output has only
copied binding values, raw-boundary labels, logs, and provenance/support status;
it borrows no pointers. There is no wait, threshold, or acceptance policy.

## Binding and availability are external provenance

View, W, and slot must carry identical source ID, epoch, generation, range ID,
prefix/slot coordinates, and fixed evidence horizon. All four IDs must be
nonzero. IDs are caller-assigned handles, not hashes, signatures, authenticated
source identities, or reliable occurrence IDs. A stale/mismatched handle fails.

`view.first_row` and `available_rows` describe a contiguous array of actually
available rows, within `capacity_rows`. `immutable_through` is an exclusive
absolute coordinate covering all consumed rows. The source is declared
append-only; W immutable and closed; and the slot sealed. Missing rows, declared
gaps, unsealed evidence, inconsistent ranges, and arithmetic overflow fail.
The clock unit is caller-defined and shared within a source/epoch/generation.
Each consumed row's actual `available_at` tick must be no later than the fixed
`evidence_horizon`; equality passes and later evidence fails with `K1_ERR_LATE`.
Zero is a valid tick, never an unknown-time sentinel. Availability times and
immutability declarations are checked for consistency, not authenticated.

All objects and their declared full buffer extents must be valid, stable,
pairwise disjoint, and inputs immutable during the call. The implementation
checks representable byte extents for overlap using `uintptr_t` addresses, on
the verified conventional flat-address host ABI. It cannot validate actual C
allocation sizes, object lifetimes, provenance truth, or concurrent writes.
Caller-supplied counts must accurately describe their buffers. The `k1_row`
layout carries exactly six contiguous floats followed by a timing field; do not
cast a plain six-float array to this struct. No shared mutable global state is
used. Distinct disjoint calls need no shared scorer storage.

Values and availability ticks outside `[prefix_begin,b)` are never inspected.
With a fixed valid binding, W, consumed rows, and horizon, appending rows after
`b` cannot change these scores. A separately later 屋 candidate cannot veto the
earlier candidate. Bounds and metadata must remain valid on every call. This
is **conditional fixed-coordinate invariance**, not proof that the coordinates
enclose a correct acoustic occurrence or that logits have local acoustic
support. A consistently wrong or ambiguous external interval can still produce
numbers; every successful result reports `K1_CALLER_BOUND_UNVERIFIED` and
`K1_FIXED_COORDINATES_ACOUSTIC_BOUNDARIES_UNPROVEN`. No result upgrades either
claim. Legacy representative peaks cannot supply W or reliable occurrence
identity. A prefix selected without the required external evidence remains an
unresolved caller task.

Re-running a model on more audio may change earlier posterior rows. This scorer
does not establish that those rows stay fixed, and its invariance claim does
not cover such a replay. The coordinates index caller-frozen posterior rows;
they are not verified word or phoneme boundaries. No model receptive-field
bound is inferred or measured by this module.

## Bounds and verification

At most **256 total consumed rows**, including W, are supported. This is an
explicit **RESEARCH capacity**, not a default latency or window recommendation.
Appended available rows may exceed it because they are not scanned. Runtime is
O(6N + 2N), N <= 256, plus constant metadata/overlap checks. Auxiliary memory is
O(1), including six DP logs and a result; there is no heap allocation or scorer
I/O. No A32, latency, throughput, real-model, or deployment claim is made.

Six named semantic test groups cover:

1. Terminal oracle: all 2,801 row matrices of lengths 0–4 from seven explicit
   dyadic distributions; **3,187,591 six-label raw paths** are exhaustively
   enumerated, including zero-product paths. A direct collapse/product/sum
   oracle checks both branch sums, joint masses, and all support statuses
2. Prefix oracle: all six entry raw states and all raw paths of lengths 0–5;
   **55,986 prefix/entry paths** check collapse, repeats, and prefix log mass
3. Named language cases: empty/all-blank, q-blank-q exclusion, contiguous q,
   mixed branches, final raw 小 versus blank, closed-prefix 小 exclusion,
   a numeric tie, no subset normalization, and entry-state repeat suppression
4. Invariance/provenance: appends containing NaNs and arbitrarily late ticks
   outside the consumed range, separate later 屋 candidate, translated/mistaken
   coordinates retaining unproven status, and output ownership
5. Probability/numerics: NaN/Inf/out-of-range values, six-class normalization,
   accepted/rejected tolerance examples, unsupported W, exact zeros, minimum
   binary32 subnormal, and finite log mass despite product underflow
6. Metadata/error atomicity: null/dimension errors, source/epoch/generation/
   range/horizon mismatch, reversed/missing/gapped/late/unsealed evidence,
   capacity and integer overflow, invalid paths, and overlapping objects

Counts describe mathematical toy enumeration, not accuracy or data coverage.
No real clips, saved posteriors, or legacy decoder data enter these tests.

Reproduce from the repository root (executables stay outside the source tree):

```sh
cd research/k1-fixed-slot-ctc-v1
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
  k1_fixed_slot_ctc.c test_k1_fixed_slot_ctc.c -lm -o /tmp/k1-test-strict
/tmp/k1-test-strict
cc -std=c11 -O1 -g -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
  -fsanitize=address,undefined -fno-omit-frame-pointer \
  k1_fixed_slot_ctc.c test_k1_fixed_slot_ctc.c -lm -o /tmp/k1-test-sanitize
ASAN_OPTIONS=detect_leaks=0 /tmp/k1-test-sanitize
```

The build/test receipt records actual results and any sanitizer environment
limitation. Tests use standard assertions; do not define `NDEBUG` for tests.
The independent oracle's command and results are in [review/REVIEW.md](review/REVIEW.md).
The path-scoped `k1-fixed-slot-math` workflow runs both strict GCC oracles; this
research source is not part of the default CMake build or installed SDK.

## Generic CTC foundation

[Graves et al., ICML 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf),
Section 3.1, equations (2) and (3), defines path products, CTC collapse, and sums
over paths yielding a labelling. That supports the generic mathematical
foundation. The externally fixed W, closed prefix run, sealed slot, shared
evidence horizon, chosen restricted terminal language, and provenance statuses
above are this module's chosen contract, not claims established by that paper.
