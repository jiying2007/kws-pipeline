# Validation report

2026-10-05, Python 3.12.14. A ZIP containing this research directory and its proposed workflow was
extracted into a new temporary directory, with no sibling research workspaces and
no PYTHONPATH. Standard-library unittest discovery passed all 59 tests. The public
report CLI and example admission CLI also returned exit 0 from that extraction.

The 59 comprise the existing 57-test guard suite (48 unit/counterexample tests plus
9 actual CLI tests) adapted to portable fixture/helper paths, and 2 new public-report
adapter tests. The separately reviewed 81-test audio/ASR-label suite is not part of
this count and is not claimed as a package test run.

Test coverage includes:

- Missing actual positives, ambiguous/incomplete labels, intent/ASR substitution,
  frozen policy mutation, duplicated PCM/WAV and invalid negative-only cohorts
- Cross-generator reference/identity leakage, exposure downgrade, unknown lineage,
  split aliases, inherited regression-only restrictions and transitive lineage
- Equal event totals hiding a new miss and false keyword; missing event arms,
  repeated/overlapping words, wrong keywords and duplicate events
- Preserved old6 detection 1/2 to 0/2, false-event recording count 1/4 to 2/4,
  and Z5/Z6 regressions, computed from the public technical event/label projection
- Invented timing examples retaining +300/+180/+120 ms without falsely claiming
  measured word-end/service latency, including missing rates and ambiguous pairing
- Fake process readers retaining null/unavailable children; missing critical RSS
  or thread observations, wrong-path ENOENT and permission denial fail closed
- Unknown supervision schemas/statuses, pre-go violations, recovery with metadata
  instead of bytes, corrupted/missing raw data and immutable failed execution status
- Admission exit codes 0/1/2, no balanced claim in exposed-regression mode, and no
  training/product/fresh-validation authorization even when requested coverage passes

Byte checks passed:

- `quality_gates.py`: unchanged SHA256
  `896641f31f997e20817e6a6ee26224d81060386b3c3583990efbb01ed3c4cd7a`
- `admit_dataset.py`: unchanged SHA256
  `d62ddbd7d10236d61306add47bc0ba2e165c3ad2ee3e209a788f5cfa307149c4`
- All six vendored supervisor definitions and four text-policy definitions exactly
  match reviewed source definition bytes; the actual-label policy whole file is unchanged
- Old6 labels/results and Dylan lineage were checked against immutable public
  GitHub source files. The public old98 summary matches the published archive member
  pin. Public pins and definition hashes are in `SOURCE_PINS.json`
- Every Python file parses. Publication review found no private source locators,
  natural-source identifiers/hashes/raw rows, machine identities, waveform/model
  files or historical executor snapshots in the proposed files

The coverage/timing/telemetry examples are deliberately invented and separately
labeled. Their test results exercise guard behavior; they do not reconstruct omitted
historical source records. The aggregate reviewed synthetic coverage summary is not
independently derivable from the invented coverage rows.

Only pure unit/CLI checks ran. Full repository C/CMake/CTest, inherited CI, model,
ASR, acoustic, native collector, board, download, installation and production-builder
integration tests were not run by this package validation. The proposed path-filtered
workflow discovers the four test files explicitly (41/7/9/2 tests verified,
including the CLI file with no standalone main) and uses the existing inventory checker
to reject an unlisted future test file. Its commands and the repository workflow
path-filter check passed locally. GitHub execution has not occurred.
The standalone API/CLI are tested; this report makes no production-integration claim.
