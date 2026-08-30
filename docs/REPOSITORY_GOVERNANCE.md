# Repository governance

This document defines repository-side controls for `kws-pipeline`. Product acoustic and physical target-board qualification remains a separate evidence boundary described in `RELEASE_QUALIFICATION.md` and Issue #2.

## Controls enforced as code

The repository carries:

- strict GCC and Clang builds plus Clang static analysis;
- C line-coverage gate, ASan/UBSan and parser fuzzing;
- Cortex-A32 ARMv7 hard-float cross-build;
- frontend parity, dataset leakage/corpus-identity audits, synthetic/domain loops, long-FAR regression and release-qualification tests;
- automatic Python test inventory so new `tests/test_*.py` files cannot silently miss Actions;
- clean CMake/pkg-config SDK consumption and two-build installed-SDK byte comparison;
- immutable-training-image contract and a real `torch_ctc` integration workflow when an immutable training image is configured;
- deterministic release archives, SHA256SUMS, SPDX SBOM and GitHub attestations;
- self-cleaning version-bound `release/vX.Y.Z` bootstrap and merged-branch cleanup;
- CODEOWNERS, product-evidence PR template and Dependabot for pinned Actions.

## Current `main` platform policy

The repository is currently in a single-maintainer/product-incubation phase. `main` is intentionally unprotected and no GitHub ruleset is installed. That is an explicit current operating choice, but **it is not called terminal governance**.

While this phase remains active:

1. steady state retains only `main`;
2. implementation/Dependabot/release-bootstrap branches are removed after use;
3. source-controlled CI is the authoritative software regression signal;
4. release creation runs the complete release workflow before publishing;
5. released `vX.Y.Z` tags are never moved;
6. direct `main` changes are permitted but cannot create a valid release without the release gates;
7. repository and release cleanup must not delete retained release/evidence history.

## Terminal governance target

Before the repository is considered fully enforced terminal product governance, GitHub platform settings should additionally require:

- a pull request for `main`;
- required checks for hosted GCC/Clang, coverage, sanitizers, fuzz and ARM cross-build;
- review-conversation resolution;
- no force-push or deletion of `main`;
- immutable release tags;
- exceptional and auditable administrator bypass.

A single-maintainer repository may omit an approving-review requirement to avoid self-deadlock. If additional maintainers are introduced, add approving/CODEOWNER review according to ownership boundaries.

These settings are GitHub administration state rather than source files and must be re-audited after ownership transfer, repository migration or policy changes. See `GOVERNANCE_TARGET.md`.

## Release policy

A formal release is valid only when the following identity is coherent:

- source commit selected for release;
- `CMakeLists.txt` SDK version;
- `vX.Y.Z` tag and GitHub Release;
- complete release CI matrix;
- two independently configured same-builder SDK installs compare byte-for-byte;
- SDK/source archives and `SHA256SUMS`;
- SPDX SBOM;
- GitHub build-provenance and SBOM attestations.

The bootstrap workflow refuses version mismatches and duplicate releases. Failed bootstrap runs delete the temporary release branch and must not leave a partial tag/Release.

## Software milestone completion

A completed software/repository milestone requires:

- no open implementation/release-maintenance PR;
- no retained implementation/release-bootstrap branch;
- green final `main` CI;
- installable SDK and reproducibility gate;
- release integrity assets when formally released;
- byte-complete training/evaluation corpus identity and machine-bound target-evidence contracts;
- explicit acknowledgement that real product measurements remain external evidence.

The absence of real Mandarin/device data does not make source CI false; it means the SKU is not acoustically/physically qualified.

## Evidence boundary

Repository CI proves software contracts, deterministic synthetic regressions, artifact relationships and build compatibility. It cannot prove real 0.3–5 m Mandarin FAR/FRR or Cortex-A32 CPU/RSS/stack/thermal/power behavior without those actual measurements. Hosted/generated evidence must never be relabeled as shipping evidence.
