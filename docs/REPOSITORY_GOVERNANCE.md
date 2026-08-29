# Repository governance

This document defines the repository-side controls for `kws-pipeline`. Product acoustic and physical target-board qualification are separate future evidence activities described in `RELEASE_QUALIFICATION.md` and Issue #2; they do not block the current software/repository terminal baseline.

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

## Current `main` policy

The current single-maintainer/product-incubation phase intentionally keeps `main` **unprotected** and does not install a branch ruleset. This is a deliberate repository policy, not a missing setup step.

The current invariants are:

1. the steady-state remote branch set contains only `main`;
2. temporary `release/vX.Y.Z`, Dependabot or implementation branches must be deleted after they are merged, closed or consumed;
3. source-controlled CI remains the authoritative software regression gate: `hosted (gcc)`, `hosted (clang)`, `sanitizers`, `fuzz` and `armv7-cross`;
4. release creation remains version-bound and runs the complete release matrix before publishing artifacts;
5. released `vX.Y.Z` tags are immutable and are never moved to a different commit;
6. direct changes to `main` are permitted in this phase, but a release must never be used to bypass CI, artifact validation, checksums, SBOM or attestations;
7. repository-cleanup and release self-cleanup must keep ephemeral branches from becoming retained product state.

If the repository later becomes multi-maintainer or externally contributed, branch protection/rulesets may be enabled then. Recommended future controls are required PRs, the five CI jobs as required checks, conversation resolution, no force-push/delete, and CODEOWNER approval. They are intentionally **not required in the current phase**.

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

## Terminal repository state

A completed repository/software milestone has:

- no open implementation/release-maintenance PR;
- no retained implementation/release bootstrap branch;
- only `main` in the steady-state branch set;
- green final `main` CI;
- reproducible installable SDK and source packages;
- checksums, SPDX SBOM and GitHub attestations for formal releases;
- no claim that hosted/synthetic evidence substitutes for real product qualification.

Real Mandarin held-out recordings, shipping microphone/enclosure/audio-pipeline qualification and physical Cortex-A32 CPU/RSS/stack/thermal/power/soak evidence may be added in a future product-validation phase. Their absence does not reopen the current software/repository milestone.

## Evidence boundary

Repository CI can prove software contracts, synthetic-control-loop behavior, artifact reproducibility and cross-build compatibility. It cannot prove production Mandarin FAR/FRR through the shipping microphone/enclosure/audio chain or physical Cortex-A32 CPU/RSS/stack/thermal/power/soak behavior. Those real-world claims remain external qualification evidence and must not be inferred from hosted CI.