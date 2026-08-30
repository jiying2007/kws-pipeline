# Changelog

All notable source-level changes are recorded here. A source/software version does **not** imply that a particular Mandarin wake-word SKU has passed acoustic or target-board qualification.

## Unreleased

- None.

## 0.3.0 software and evidence hardening — 2026-08-30

### Runtime and integration

- Added exact runtime build identity, including source revision, target/config digest and dirty-source marking.
- Added versioned discontinuity/external-AFE metadata and v2 runtime telemetry.
- Added `kws_engine_notify_discontinuity()` with XRUN/route/clock/suspend-resume reasons so missing audio cannot silently bridge acoustic state.
- Preserved the deployable **KWSP ABI v2** and **KWKP ABI v3** while hard-cutting the public software/evidence contracts.

### Corpus, training and evaluation integrity

- Added canonical training/evaluation corpus identity with per-WAV file SHA256, decoded mono-16-kHz PCM16 SHA256, frame count/duration and whole-corpus SHA256.
- Added TSV + schema-rich JSONL training support with speaker/session/source/room/device metadata.
- Model provenance **schema v3** now binds the actual training corpus rather than only manifest bytes.
- Evaluation provenance **schema v2** binds every held-out WAV and rejects `references.duration_s` that disagrees with real WAV duration.
- Qualification requires a clean dataset-audit artifact covering the exact selected training manifests and final references file with speaker/session/source metadata for human data.
- Restored Python 3.8 compatibility for synthetic-domain generation without weakening the v0.3 evidence contracts.

### Product-board evidence

- Target evidence **schema v2** is hard-cut to `evidence_class=product-board` for shipping resource evidence.
- Product-board evidence now binds SKU, exact source SHA, builder/DUT/collector identities, collector hash, board runner, model, keyword pack and board audio.
- Builder and DUT identities must be distinct.
- Added canonical `evidence-raw.jsonl` identity: exact `{name, sha256, bytes}` set equality for runtime-soak, power and all additional raw measurement artifacts.
- Added external attestation-verification **schema v1** binding the canonical raw-evidence manifest, collector, board runner, model and keyword pack to an approved trust-policy result.
- Runtime-soak CPU/RSS/thermal summaries are independently recomputed from retained raw samples; summary-only tampering is rejected.
- Raw power evidence requires original instrument output plus instrument/calibration identity.
- Target board benchmark output now binds runtime source/config identity.

### Qualification and policy

- Qualification manifest **schema v2** reopens/re-hashes original training/evaluation WAVs and validates product-board raw/attestation identities.
- Qualification policy **schema v2** identifies the SKU and requires `shipping_approved=true` for a shipping pass.
- Qualification gate result **schema v3** binds the exact manifest/policy and applies FAR/FRR confidence bounds plus latency/resource/soak/power gates.
- Final FAR exposure is derived from actual WAV frames and cannot be inflated by a reference-only duration declaration.

### Training supply chain

- `training/Dockerfile` no longer resolves/upgrades dependencies from the network and requires a digest-pinned immutable OCI base.
- Added `training/build_container.py` for controlled wrapper-image construction and build receipts.
- Shipping checkpoints can require the final immutable training image digest.
- Added manual/weekly real `torch_ctc` end-to-end integration driven by digest-pinned repository variable `KWS_TRAINING_IMAGE`.

### CI, release and reproducibility

- Added Clang static analyzer to CI/release and generated-build-header analysis support.
- Added C line-coverage gate.
- Added ASan/UBSan and `.kwm/.kwk` libFuzzer release gates.
- Added Cortex-A32 ARMv7 hard-float cross-build gate.
- Added Python test-inventory enforcement so new `tests/test_*.py` cannot silently remain outside official workflows.
- Added two independent SDK builds with byte-for-byte installed-tree comparison.
- Release publishing requires the full hosted/coverage/sanitizer/fuzz/Cortex-A32/reproducibility matrix and emits deterministic SDK/source archives, SHA256SUMS, SPDX SBOM and GitHub attestations.
- Documentation examples are tested against the final v0.3 product-board CLI contract.

### Evidence boundary

v0.3 closes the identified **software/evidence-engineering** gaps. It still does not manufacture final product evidence. Independent real Mandarin held-out speakers through the shipping microphone/enclosure/audio-pipeline, genuine 0.3–5 m acoustic coverage and physical Cortex-A32 CPU/RSS/stack/thermal/power/soak measurements remain Issue #2 gates.

## 0.2.0 software baseline — 2026-08-29

- Added KWKP ABI v3 shared-prefix arbitration, per-keyword policy/priority/trailing blank, dual logmel/PCEN-lite frontend support and frontend-spec v2 lineage.
- Added acoustic-scene rendering, domain metrics/curriculum, long-FAR regression, statistical FAR/FRR confidence bounds and identity-aware dataset audit.
- Added training-environment provenance, target board benchmark, deterministic release packaging, SPDX/SHA256/attestations and parser fuzzing.
- Software/runtime/synthetic/release-integrity mechanisms were established, but original audio bytes and trusted product-board evidence were not yet fully bound; v0.3 closes those software gaps.

## 0.1.0 source baseline — 2026-08-29

- Fixed-memory C11 always-on KWS runtime.
- 16-kHz / 25-ms / 20-ms-hop frontend and tiny int8-weight streaming recurrent acoustic model.
- KWSP ABI v2 model format and vocabulary-bound field-updatable keyword packs.
- Shared-prefix keyword trie and configurable Mandarin/pinyin wake phrases.
- L0 keyword update, L1 calibration/hard-negative workflow and L2 `--head-only` shallow customization.
- CTC training/export reference toolchain, continuous-audio evaluation, strict hosted CI and installable CMake/pkg-config SDK.

The source baseline was never a claim that a real Mandarin base model or target-board acoustic qualification had been completed.
