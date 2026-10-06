# Independent review: K1 fixed-slot CTC research v1

Date: 2026-10-06 UTC. Verdict: **PASS for the declared pure mathematical
contract, with external provenance and acoustic boundaries unproved.** No
remaining implementation defect was found in this review. This is neither an
acoustic validation nor approval to integrate or deploy the scorer.

## Scope and resolved finding

Reviewed the frozen public API, implementation, README, tests, and build receipt.
The initial API lacked a fixed availability horizon, so a consumed row produced
late could not be distinguished from evidence available on time. The owner fixed
this before freeze: all bindings include `evidence_horizon`; each consumed row
has `available_at`; a newer row returns `K1_ERR_LATE`. Independent tests verified
late prefix and terminal rows, equality at the horizon, and ignored late rows
outside the consumed interval.

The two three-state recurrences sum exactly `blank* q+ blank*`, with q respectively
窝 and 屋. They share the closed external W and sealed `[a,b)` and do not select a
window or decision. Prefix validation applies the declared previous raw symbol
before CTC collapse and requires exactly the additional `你好小`. Continued 小
inside the closed slot and q-blank-q are excluded. All six supplied FP32 values
are checked; none are redistributed or zero-floored. The documented 1e-5 row-sum
tolerance admits literal slightly nonunit path masses, including positive log
masses; these are not clamped or claimed to be exactly normalized probabilities.
Log arithmetic uses binary64 and handles zero support without underflowing a
positive path to the zero-support status.

Range, capacity, integer-overflow, overlap, missing/gapped/late evidence, sealing,
prefix support, and binding checks precede output assignment. Output owns copied
values; errors leave its bytes untouched. Declared C object sizes, immutability,
source identity, and clocks remain the caller's responsibility. Checked equal
handles are neither cryptographic authentication nor acoustic occurrence proof.

## Independent execution

The accompanying `independent_oracle.c` was written from the public API before
reading the owner's oracle. It independently enumerates raw six-label paths,
applies CTC collapse, and sums integer products over powers of eight. It does not
reuse the scorer's recurrence or states. The finite family contains six one-hot
rows and two explicit dyadic mixtures. For every sequence of lengths 0–3:

- 585 row matrices, exactly `1 + 8 + 8^2 + 8^3`
- 1,170 branch-mass comparisons
- 112,945 raw six-label paths, exactly `1 + 48 + 48^2 + 48^3`
- 7 additional successful calls covering closed-Xiao and repeated-q exclusion,
  a nonunit prefix product, nonblank entry state, trailing prefix blank, and
  fixed-posterior append invariance
- 22 expected-error calls covering numerical, binding, timing, range, capacity,
  overflow, prefix, and overlap failures, with unchanged input/output bytes

GCC 14.2.0 strict C11 build and this independent test execution passed after the
owner declared terminal freeze. Command from the repository root:

```sh
cc -std=c11 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror \
  -I research/k1-fixed-slot-ctc-v1 \
  research/k1-fixed-slot-ctc-v1/k1_fixed_slot_ctc.c \
  research/k1-fixed-slot-ctc-v1/review/independent_oracle.c -lm \
  -o /tmp/k1-independent-review
/tmp/k1-independent-review
```

Observed output:

```text
exhaustive: 585 row sequences, 1170 branch masses, 112945 full six-label paths
directed: 7 successful calls and 22 atomic error checks
independent fixed-slot CTC review tests: PASS
```

The owner's separate suite was inspected, not rerun by this reviewer. Its oracle
also enumerates complete raw paths and direct collapse/product/sum. Its dyadic
finite support permits exact binary64 path products and sums at the tested small
lengths. Its count assertions match 2,801 matrices, 3,187,591 terminal raw paths,
and 55,986 prefix/entry paths. The frozen owner receipt reports strict and
ASan/UBSan passes with leak checking disabled. LeakSanitizer is explicitly
**blocked** by the environment's ptrace limitation, not passed. Independent
sanitizer execution was not repeated.

## Limits carried forward

- K2 is outside this implementation. No result establishes K2 coverage or
  changes an external occurrence-association status
- W, prefix closure, and the terminal cell are external choices. Their
  correspondence to a spoken occurrence remains unproved; new acoustic labels
  are not required to establish this algorithmic contract
- Append invariance holds only when the already-consumed posterior rows remain
  fixed. Re-running a model on more audio may change them
- Tests use invented probabilities only. No model, decoder, ASR, audio, TTS,
  training, saved-data replay, parameter sweep, or network publication was run

## SHA-256 binding

The following frozen hashes were independently checked before test execution.
Paths are relative to `research/k1-fixed-slot-ctc-v1`. The README hash identifies
the pre-publication reviewed README; the public README adds packaging and test-path
clarifications. The C/header/oracle/receipt bytes remain unchanged.

```text
70dfaf5702236a7fbe35195f0ac970aea8f29e1113d2ce2a918bfc9bff2c37e3  k1_fixed_slot_ctc.h
fcb7d3cac5054a7594e30c6542f1b6786fe8b8e987e61d06954604ec6dff116d  k1_fixed_slot_ctc.c
477c08ada361d420cc1c093597360fbe6f1ea45da741f6e0872be2262e40eabf  test_k1_fixed_slot_ctc.c
8f8cbc65540576cc8592498899fd8592c1583d8c9ffadd53863f66acf4cf857f  README.md
fda6de86496f0973f2f19f716863d6667098fef027f358af931fcef12be7cfc5  RECEIPT.txt
aac165319240f22b7e025e108a09638b88ee984b5f5b6ddd06b2be15135c9f74  review/independent_oracle.c
```
