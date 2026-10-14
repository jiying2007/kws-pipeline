# Test declarations and actual execution evidence

`tools/test_inventory.py` inventories executable Python commands in workflow
`run` blocks. Comments, path filters, echo statements, heredoc bodies and test
filenames passed as data to other programs do not satisfy the inventory. Its
output explicitly distinguishes mainline, conditional, path-filtered,
manual/scheduled, release-only and hardware/self-hosted declarations. A selected
class/script invocation is also identified separately. This is a wiring check,
not execution, statement/branch coverage, or proof that every test ran.

The official inventory currently includes tests whose only routes are optional
training or path-filtered jobs. A green ordinary mainline CI does not certify
those jobs, real-model inference, private corpus qualification or physical-board
acceptance. Historical totals such as 5,246 refer only to the particular recorded
run and command set, not all repository tests or all environments.

## Guarded unittest suites

`tools/test_execution_contracts.json` records the reviewed exact discovery count,
SHA-256 of sorted unittest case IDs, three required Python modes and exact allowed
skip ID/reason map for each guarded suite. `tools/checked_unittest.py` rejects zero
or changed discovery, duplicate IDs, partial execution, omitted start/stop
callbacks, unexpected skips, stale allowed skips, errors, failures, expected
failures and unexpected successes. The default skip map is empty.

Reports contain the actual discovered, started, finished and successful case IDs,
counts, skip reasons and outcomes, Python mode/version, contract digest and
GitHub source/run/attempt/job identity. All normal/-O/-OO reports must validate
before the job passes. Reports are uploaded as `checked-unittest-python-contracts`
and `checked-unittest-posterior`. These artifacts cover only the named guarded
suites; other script-style and C tests remain separate checks. Reports are local
execution evidence, not an independent cryptographic attestation.

A test addition/removal/rename must update its reviewed count and ID digest in
the same change. The runner deliberately has no automatic baseline-rewrite mode.
Do not accept a new count merely to make a failed CI green: review the discovered
ID change and retain all intended cases and modes.

## Parallel compiler work and required status names

`hosted-build (gcc)` and `hosted-build (clang)` start independently of Python and
research checks. Both retain native build/CTest, real compiler artifacts, replay,
parity, SDK install/consumer and reproducibility checks. Compiler-independent
promotion admission/source suites run once in each Python mode, outside that
matrix.

The existing `hosted (gcc)` and `hosted (clang)` required status names are retained
as `always()` aggregate jobs. Every unconditional dependency must be `success`;
failure, cancellation, missing outputs and unexpected skips fail closed. The
research job may be `skipped` only when its validated selector is exactly `false`.
No branch/ruleset setting is changed by this refactor.
