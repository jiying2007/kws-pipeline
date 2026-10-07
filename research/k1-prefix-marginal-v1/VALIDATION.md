# Validation and preservation

The relocated public package was checked on 2026-10-07 with Python 3.12.
Only synthetic/adapter checks ran during publication preparation. No real
saved inputs were reopened or evaluated.

| Check | Observed result |
| --- | --- |
| Mathematical module unittest suite | 30 passed |
| Original saved runner mock suite | 7 passed |
| Recovery runner synthetic/adapter suite | 12 passed |
| Independent literal raw-path oracle | 612 cases, 5,523,157 paths, zero failures |
| Exact oracle EOF certificates | 20 accepts, 586 rejects, 6 unresolved ties |
| Independent API/numerical/lifecycle checks | 11 groups passed, 5,184 additional exact raw paths |
| Independent underflow adapter checks | 8 groups passed, no saved values or DP runs |

The original tie-failure report remains at `review/initial-oracle-results.json`.
The corrected mathematical implementation fails closed for unresolved exact
ties, including the originally false-accepted alignment tie. The historical
original softmax adapter and its failed saved-run ledger are preserved. The
recovery's numerical change is limited to already-rounded zeros; the semantic
gate remains unchanged and the candidate still fails.

All 28 directly copied source/evidence files match their preparation originals
byte for byte. This includes the entire current module and test suite, both
frozen runners/tests/protocols/manifests/release templates, three independent
review programs, their retained reports, the initial failed ledger and cause
diagnosis, and final recovery results/ledger. Review-manifest projections
replace local paths with logical source identifiers while retaining scientific
hashes. `SOURCE_MAP.json` maps those identifiers to the public repository files;
`review/projection-map.json` records the projection identities.

The saved-result audit was receipt-only: it checked exact rational
inequalities, order, counts, and frozen hashes without reproducing the DP.
That audit passes as an evidence-integrity check. It does not turn the failed
semantic gate into a pass.

The new workflow is path-scoped, read-only, uses a pinned official checkout,
installs no dependencies, and runs the same offline synthetic checks. The two
public release requests remain `NOT_RELEASED`. The publication preparation
contains no actual execution release, raw input, audio, model or dependency
payload. No production code or default setting is modified.

This document records local pre-publication checks. Remote GitHub CI and the
full repository C/build/sanitizer/board suite have not been run as part of this
preparation. Any later CI outcome must be reported for its exact remote commit.
