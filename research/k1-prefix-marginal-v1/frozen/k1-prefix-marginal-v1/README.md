# K1 first-prefix CTC marginal, research v1

Standalone, dependency-free Python implementation of the supplied mathematical
design. It marginalizes every CTC path using ten logical states and makes one
decision at explicitly supplied EOF. It does not replace a live decoder.

This directory contains code and synthetic-only tests. Its preparation involved
no audio, real saved logits/posteriors, model inference, training, live decoder
evaluation, network access, or publication. The real M3 and N1 values have not
been read by this implementation task. Synthetic examples named for their
semantic roles are not evaluations of those observations.

## Fixed semantics

The complete alphabet, in exact input-column order, is:

```python
TOKENS = ("blank", "你", "好", "小", "窝", "屋")
```

CTC first merges adjacent identical raw symbols and then removes blank. Thus
`你 你` emits one `你`, while `你 blank 你` emits two. Whole-observation classes are:

- A: collapsed transcript starts with `你好小窝`
- B: collapsed transcript starts with `你好小屋`
- R: all other paths, including incomplete prefixes and incompatible starts

These disjoint exhaustive masses are never conditionally renormalized. A, B,
and the mismatch state are absorbing; all future suffix rows are marginalized.
Leading blank is allowed. Unrelated speech before a wake is not a positive
first-prefix example. This is not a keyword-anywhere or continuous detector.

The mathematical EOF rule is `ACCEPT_K1 iff A > B + R`; ties reject. Since the
exact masses sum to one, this is equivalent to `A > 1/2`. This is a predeclared
equal-cost binary model decision, not an empirically calibrated wake threshold.
Model-implied CTC masses need not be calibrated semantic class probabilities.

## Input contract

Each row supplies exactly six ordered, finite, nonnegative literal weights,
with at least one positive weight. Every row is normalized over all six tokens.
Explicit zeros, including negative floating zero, are legal. A common positive
scale factor leaves the mathematical result unchanged; binary64 conversion of
arbitrary scaled inputs can add rounding. Exact representable power-of-two
scaling has the same certified bounds and is covered by tests.

The accepted numeric contract is Python real numbers converted to binary64
`float`. Certification concerns those converted binary64 values. Booleans,
strings, complex values, nonfinite/negative weights, nonrepresentable finite
values, and nonzero inputs that underflow on conversion are rejected. The row
normalizer does not form an overflowing floating sum or divide a tiny numerator
in linear space: it forms centered log weights and a stable six-way log sum.
Logs are natural logs. The separate certification calculation normalizes the
same six binary64 weights using their exact integer ratios.

Constructor `token_order` must equal the declared tuple exactly, including its
blank spelling. Missing, added, duplicated, or permuted columns are errors.
Ordered iterables are accepted, but mappings, sets, and strings are rejected.
A row reads at most seven values, so an unbounded row iterable fails promptly.

No logits converter is included. Do not pass raw logits, decoder scores, a
selected subset of a larger head, or values whose provenance is unknown. A
future separately declared unit-temperature softmax adapter would need to state
that its binary64 exponential outputs are literal weights; their mathematical
transcendental values would not be certified by this module.

## API and finite-observation lifecycle

Requires Python 3.10+ with binary64 floats; verified here on Python 3.12.14.

```python
from k1_prefix_marginal import K1PrefixMarginal, TOKENS

verifier = K1PrefixMarginal(token_order=TOKENS)
for token in ("你", "好", "小", "窝"):
    verifier.update([int(column == token) for column in TOKENS])
result = verifier.finalize()  # The caller supplies EOF, not a guessed endpoint.
assert result.decision == "ACCEPT_K1"
assert result.decision_status == "CERTIFIED_ACCEPT"
assert result.masses == (1.0, 0.0, 0.0)
assert result.emitted
assert not verifier.finalize().emitted
```

- `update(weights)` consumes one complete row and returns a snapshot
- `update_chunk(rows)` consumes an ordered iterable of rows and returns a snapshot
- `snapshot()` returns an immutable readout without deciding or changing state
- `finalize()` returns an immutable EOF result; it inserts no row or blank
- `reset()` explicitly begins a new observation and returns its empty snapshot

Chunk boundaries insert no blank or reset. Empty chunks are no-ops before EOF.
Both single-row and whole-chunk updates are atomic for the verifier: malformed
input, iterator failure, or capacity overflow leaves its prior state intact.
The external iterator's consumption cannot be rolled back. No input history is
buffered. The object is not thread-safe; callers must serialize its operations.

Every observation has an explicit research limit of `MAX_FRAMES = 4096`,
counting all rows, including silence. This is not an acoustic window and does
not trigger a reset. Going beyond the limit raises `CapacityError`; a new
observation requires an explicit external boundary and `reset()`. Never divide
an over-limit observation into resets to manufacture acceptance.

EOF emits once: the first result has `emitted=True`, and repeated finalization
preserves the same snapshot/decision/status with `emitted=False`. Updates after
EOF, including empty chunks, raise `FinalizedError` until reset. The caller
should deliver an event only when `emitted` is true. `reset()` after either a
partial or finalized observation clears all old state and the emission flag.

## Readouts and numerical certificates

Snapshots expose `frames`, `finalized`, `A`, `B`, `R`, `log_A`, `log_B`, `log_R`,
the tuple conveniences `masses` and `log_masses`, and ten state masses/logs in
`STATE_NAMES = (e,b1,r1,b2,r2,b3,r3,A,B,D)` order. The state logs use stable
float64 log addition and direct nonnegative mismatch flux. They retain finite
log probabilities even when rendering their linear masses yields zero. No
bucket clipping, subtraction from one, or bucket renormalization hides error.
Both the log and linear readouts are approximations, not certified values.

A strict comparison of approximate logs can accept an exact tie. For example,
after deterministic `你 好 小`, rows `{blank:4, 窝:1}` and `{blank:5, 窝:3}` give
exact `A = 1 - (4/5)(5/8) = 1/2`, but common floating evaluation gives a slightly
larger log A than log R. This is a regression test.

To keep such ties from accepting, each of the same ten logical states also has
an exact integer lower/upper enclosure on a fixed `2^-256` probability grid.
`CERTIFICATE_BITS = 256` is arithmetic precision, not a fitted decision epsilon.
There is no adaptive precision or threshold search. Exact ratios of the input
binary64 values give integer row weights. Integer multiplication/addition
evaluates the same nonnegative linear map, followed by floor for each lower
output and ceiling for each upper output. Absorbing states use the row's exact
unit continuation mass. Arbitrary-precision Python integers prevent arithmetic
overflow in this bounded calculation.

EOF results expose `decision_status` in addition to the binary `decision`:

- `CERTIFIED_ACCEPT`: the exact A lower bound is strictly above `1/2`
- `CERTIFIED_REJECT`: the exact A upper bound is at most `1/2`
- `NUMERICALLY_UNRESOLVED`: the interval straddles that boundary; the operational
  decision is `REJECT_K1` (no acceptance), but this is not a proven mathematical
  negative. An exact tie or an extremely small positive margin can be unresolved

The exhaustive-partition identity makes this certificate equivalent to the
fixed mathematical comparison `A > B + R`; R has not been omitted or split.
Certification is solely about the model path distribution defined by literal
binary64 input weights. It proves neither acoustic truth nor model calibration.

Every snapshot includes `state_lower_numerators`, `state_upper_numerators`, and
`mass_bounds`, with the latter returning exact `Fraction` lower/upper pairs for
A, B, and R. Divide state numerators by module constant `CERTIFICATE_SCALE`.
EOF results include that full snapshot and mass/log tuple conveniences. Keep
bound numerators and denominators when serializing: converting a narrow bound
to two floating numbers can falsely display an exact tie. Validation of a
semantic negative must require `CERTIFIED_REJECT`, not merely `REJECT_K1`; an
unresolved result is inconclusive for either semantic class.

## Enclosure proof and resources

Each row's ten-state transition matrix is nonnegative and column-stochastic.
Applying it to a lower vector or upper vector preserves the ordering around
the exact state. Downward/upward rounding each resulting component therefore
preserves enclosure by induction from the exact empty state. The total lower
deficit increases by less than ten grid units per row; the total upper excess
obeys the same bound. At T rows:

- each one-sided L1 error is at most `10*T/2^256`
- the sum of full enclosure widths is at most `20*T/2^256`

These are guaranteed probability-enclosure bounds, not a decision margin.
The generic bound is conservative; exact deterministic/representable paths can
have zero-width intervals. With T at most 4096, lower state numerators fit in
257 bits and upper numerators are at most `2^256 + 40960`, also 257 bits. Exact
binary64 row ratios have bounded few-thousand-bit integer intermediates.

Persistent numeric state is ten binary64 log values and twenty bounded integer
endpoints, plus row count and lifecycle metadata. This implements ten logical
states, not a claim of only ten physical scalar fields. Input conversion and
snapshot creation use bounded temporary storage. Work is O(T) and live memory
is O(1) for this fixed alphabet/prefix and 4096-row research limit. Caller-held
snapshots or input arrays are outside the module's constant-memory claim.

## Known semantic limitations

After `你 好 小`, the rows `{窝:3, blank:2}` then deterministic `屋` produce
`(A,B,R)=(3/5,2/5,0)` and accept. Clear later 屋 cannot retract the already
absorbed 窝 branch. This is a documented counterexample to a confuser-repair
claim, not a passing protection result.

A genuine `你好小窝`, followed by a blank and an independent later `屋`, remains
in A under every suffix. This desired suffix protection and the preceding
limitation are consequences of the same first-prefix semantics. Identical
posterior inputs cannot reveal whether later evidence refers to the same
spoken occurrence or a new one. A blank gap alone is not a linguistic boundary.

No onset/end time, endpoint quality, latency, duplicate-event policy, broader
vocabulary, held-out accuracy, or live-decoder readiness is established here.
Any saved-data evaluation, inference, integration, or deployment is a separate
task and is not performed by this module or these synthetic tests.

## Verification

From this directory:

```sh
python -m unittest discover -s tests -v
python -m compileall -q k1_prefix_marginal.py tests
```

The standard-library suite covers all 17 supplied directed fixtures; every
two-way split; row-by-row and deterministic random ragged chunks; adjacent and
blank-separated repeats; arbitrary absorbed suffixes; true-prefix protection
and the irreversible-accept counterexample; reset/EOF semantics; transactional
validation/iterator/capacity failures; extreme and subnormal weights; tiny
positive paths over 4096 frames; conservation, monotonicity, fixed-grid error
budgets; exact and multi-alignment ties; adjacent binary64 weights on either
side of a tie; and an explicitly unresolved subgrid positive margin.

The separate independent review enumerates literal raw paths, merges raw
repeats before removing blank, and uses literal collapsed-prefix checks. It
does not share this optimized recurrence. Review code/results are maintained
separately from this implementation directory.

Design basis: supplied `prefix-marginal-feasibility-v1/design.md` and
`fixtures.md`. Standard CTC path/collapse/forward definitions originate in
[Graves et al., 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf); the
specialized ten-state recurrence, equal-cost EOF rule, and directed interval
certificate are the research construction documented here. No network access
was used for implementation or synthetic verification.
