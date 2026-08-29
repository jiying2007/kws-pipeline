# Repository governance

This document defines the repository-side controls for `kws-pipeline`. Product acoustic and physical target-board qualification are separate evidence gates described in `RELEASE_QUALIFICATION.md` and Issue #2.

## Enforced in the repository

The repository carries the following controls as code:

- strict GCC and Clang CI;
- ASan/UBSan and parser fuzzing;
- Cortex-A32 ARMv7 hard-float cross-build;
- frontend parity, dataset leakage audit, synthetic/domain training loop, long-FAR smoke and release qualification tests;
- clean CMake/pkg-config SDK consumer validation;
- deterministic release archives, SHA256SUMS, SPDX SBOM and GitHub attestations;
- a self-cleaning version-bound `release/vX.Y.Z` bootstrap for environments that cannot create tag refs directly;
- merged-branch cleanup with race rechecks;
- CODEOWNERS and a product-evidence PR checklist;
- Dependabot tracking for pinned GitHub Actions dependencies.

## Required GitHub repository settings

The intended `main` policy is:

1. require a pull request before merging;
2. require the five CI jobs `hosted (gcc)`, `hosted (clang)`, `sanitizers`, `fuzz` and `armv7-cross` to succeed;
3. block force pushes and branch deletion for `main`;
4. require conversation resolution before merge when review threads exist;
5. prefer squash merge for the linear product history;
6. keep administrator bypass exceptional and auditable;
7. never move a released `vX.Y.Z` tag to a different commit.

On a single-maintainer personal repository, requiring an approving review from the PR author is not workable. The primary invariant is therefore **PR + required CI + no force-push/delete**. If a second maintainer is added, enable at least one approving review and CODEOWNER review.

Repository settings are platform administration state, not source files. They must be checked after repository migration, ownership transfer or GitHub policy changes.

## Release policy

A release is valid only when all of the following refer to the same commit:

- `main` source tree chosen for release;
- SDK version in `CMakeLists.txt`;
- `vX.Y.Z` tag;
- GitHub Release;
- SDK/source archives and their `SHA256SUMS`;
- SPDX SBOM;
- build-provenance and SBOM attestations.

The release workflow refuses version mismatches and duplicate releases. A failed bootstrap run deletes its temporary `release/vX.Y.Z` branch and must not leave a partial tag or GitHub Release.

## Evidence boundary

Repository CI can prove software contracts, synthetic-control-loop behavior, artifact reproducibility and cross-build compatibility. It cannot prove production Mandarin FAR/FRR through the shipping microphone/enclosure/audio chain or physical Cortex-A32 CPU/RSS/stack/thermal/power/soak behavior. Those real-world claims remain external qualification evidence and must not be inferred from hosted CI.
