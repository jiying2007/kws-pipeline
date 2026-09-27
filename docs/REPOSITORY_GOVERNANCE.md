# Repository governance

This document defines repository-side controls for `kws-pipeline`. Product acoustic and physical target-board qualification remains a separate evidence boundary described in `RELEASE_QUALIFICATION.md` and Issue #2.

## Controls enforced as code

The repository carries:

- strict GCC and Clang builds plus Clang static analysis;
- C line-coverage gate, ASan/UBSan and parser fuzzing;
- a symbol-level purity gate on `libkws_pipeline.a` so the "no heap, hidden thread, lock, filesystem or text/pinyin conversion" runtime claim cannot regress silently;
- Cortex-A32 ARMv7 hard-float cross-build whose CTest suite is actually executed under `qemu-arm-static`, plus a hosted-vs-Cortex-A32 numerical parity gate on the int8 inference kernel;
- frontend parity, dataset leakage/corpus-identity audits, synthetic/domain loops, long-FAR regression and release-qualification tests;
- automatic Python test inventory so new `tests/test_*.py` files cannot silently miss Actions;
- clean CMake/pkg-config SDK consumption and two-build installed-SDK byte comparison;
- immutable-training-image contract and a real `torch_ctc` integration workflow when an immutable training image is configured;
- deterministic release archives, SHA256SUMS, SPDX SBOM and GitHub attestations, plus `tools/verify_model_release.py` so a downloaded model release is checked against the contract pins rather than only against the manifest shipped beside it;
- self-cleaning version-bound `release/vX.Y.Z` bootstrap and merged-branch cleanup;
- CODEOWNERS, product-evidence PR template and Dependabot for pinned Actions.

## Current `main` platform policy

As of 2026-09-27, the live GitHub repository ruleset `kws-main-terminal` is
**active** on the default branch and the branch API reports `main` as
`protected=true`. The reviewed machine target is
`governance/main-ruleset-target.json`, and
`governance/verify_live_main_ruleset.py` verifies live ruleset JSON against
that source-controlled target.

The enforced ruleset currently requires:

1. updates to `main` through a pull request;
2. all review conversations resolved;
3. squash as the only merge method;
4. strict required checks `hosted (gcc)`, `hosted (clang)`, `coverage`,
   `sanitizers`, `fuzz`, and `armv7-cross`;
5. deletion protection;
6. non-fast-forward protection, blocking force pushes;
7. no bypass actors.

The approving-review count remains zero intentionally while the repository is
single-maintainer, avoiding self-deadlock without weakening the other PR and
status-check requirements.

GitHub's classic branch-protection sub-object may still report
`protection.enabled=false`; enforcement for this repository comes from the
repository ruleset above. Do not interpret the classic endpoint alone as
evidence that `main` is unprotected.

Steady-state branch cleanup, immutable release/tag policy, source-controlled CI
and retained evidence history remain part of the repository contract.

## Governance re-audit boundary

GitHub rulesets are platform state, not Git tree state. Re-read and verify the
live ruleset after repository migration, ownership transfer, administrator
changes or any governance-policy update. If additional maintainers are
introduced, review the zero-approval single-maintainer choice and add
approving/CODEOWNER requirements as appropriate rather than weakening the
existing terminal rules.

The source-controlled target and verifier are recovery/audit inputs; they do not
grant permission to bypass live platform enforcement.

## Retired execution-lane policy

Closed research execution lanes are not compatibility APIs. Once a lane has a retained
machine-readable closure/evidence record and the live product authority has moved to a
canonical owner, its old workflows, helpers, mirror tests and compatibility entry points
must stay absent.

`tests/test_repository_cleanup_contract.py` enforces this for the retired
generalization, GRU/RNN parallel-development, startup/preceding-context, generic synthetic
loop and superseded decoder-diagnostic paths. It also verifies that the canonical product
training, qualification and shipping authorities remain present.

If future product data justifies reopening one of those research questions, introduce an
explicit new lane with fresh scope and evidence boundaries. Do not silently resurrect the
retired path names or re-add mirror implementations merely for compatibility.

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
