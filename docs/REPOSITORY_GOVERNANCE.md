# Repository governance

This document defines repository-side controls for the **research-only** `kws-pipeline`. Independent human and physical target acoustic/resource measurements are scientific evidence, not commercial approval or a requirement to publish a model. Historical Release/Shipping documents remain provenance only; current research authority is [research-only execution](RESEARCH_ONLY_EXECUTION.md).

## Controls enforced as code

The repository carries:

- strict GCC and Clang builds plus Clang static analysis;
- C line-coverage gate, ASan/UBSan and parser fuzzing;
- a symbol-level purity gate on `libkws_pipeline.a` so the "no heap, hidden thread, lock, filesystem or text/pinyin conversion" runtime claim cannot regress silently;
- Cortex-A32 ARMv7 hard-float cross-build whose CTest suite is actually executed under `qemu-arm-static`, plus a hosted-vs-Cortex-A32 numerical parity gate on the int8 inference kernel;
- frontend parity, dataset leakage/corpus-identity audits, synthetic/domain loops, independent long-FAR regression and retained historical qualification negatives;
- automatic Python test inventory so new `tests/test_*.py` files cannot silently miss Actions;
- clean CMake/pkg-config SDK consumption and two-build installed-SDK byte comparison;
- immutable-training-image contract and a real `torch_ctc` integration workflow when an immutable training image is configured;
- deterministic source/model SHA256 manifests and retained historical release/SBOM attestations for audit; source-only tests must never grant publication authority;
- source-bound research branch cleanup, with historical release bootstrap records preserved only in Git;
- CODEOWNERS, research evidence PR templates and Dependabot for pinned Actions.

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

Steady-state branch cleanup, immutable historical tags/archives, source-controlled
research CI and retained evidence history remain part of the repository contract.

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
canonical research owner, its old workflows, helpers, mirror tests and compatibility entry points
must stay absent.

`tests/test_repository_cleanup_contract.py` enforces this for the retired
generalization, GRU/RNN parallel-development, startup/preceding-context, generic synthetic
loop and superseded decoder-diagnostic paths. It also verifies that the canonical research training, negative evaluation and source-history authorities remain present.

If future permitted research data justifies reopening one of those scientific questions, introduce an
explicit new lane with fresh scope and evidence boundaries. Do not silently resurrect the
retired path names or re-add mirror implementations merely for compatibility.

## Historical release records (not executable research)

The former `release.yml`, `deployment-release.yml`, `model-promotion.yml`,
`shipping-approval.yml` and manual human/DUT release publishers have been
removed from active Actions. The now-unreferenced `tools/verify_release_tag.py`
bootstrap helper is also retired; its original source and annotated-tag
resolution behavior remain available at the prior Git commit for audit, not
as a callable research entrypoint. Nothing here authorizes creating, moving,
or deleting historic immutable tags, Releases or model evidence.

Current PR/main research CI checks source/rights, data roles, reproducibility,
model registry SHA, acoustic and resource evidence and negative cases. It does
not require a product release tag, SDK publication, commercial approval, or a
success-labelled physical qualification. A new research experiment must be
separately authorized and hash-bound; older synthetic failures retain their
original FAILED state.

## Software milestone completion

A completed software/repository milestone requires:

- no open unclassified research implementation PR;
- no retained unclassified research execution branch;
- green final `main` CI;
- installable SDK and reproducibility gate;
- source/rights and immutable research evidence inventories with truthful failures;
- byte-complete training/evaluation corpus identity and machine-bound target-evidence contracts;
- explicit acknowledgement that real human/physical-target measurements remain independent and external.

The absence of real Mandarin/device data does not make source CI false; it means independent acoustic/physical results are NOT_RUN, not PASS.

## Evidence boundary

Repository CI proves software contracts, deterministic synthetic regressions, artifact relationships and build compatibility. It cannot prove real 0.3–5 m Mandarin FAR/FRR or Cortex-A32 CPU/RSS/stack/thermal/power behavior without those actual measurements. Hosted/generated evidence must never be relabeled as real-person or real-device evidence.
