# Changelog

All notable source-level changes are recorded here. A source/software version does **not** imply that a particular Mandarin wake-word SKU has passed acoustic or target-board qualification.

## Unreleased

- Resolve fully tied keyword candidates by stable keyword ID rather than input order. Rank all pending candidates within each frame before updating the held winner, preserving its grace age when it remains the winner even as confidence improves. Existing policy-specific priority/depth/confidence ordering and all model/threshold values are retained. Historical acoustic qualification is not transferred to this new runner.
- Refresh automatically detected Git source revision and dirty state during incremental builds, including builds without an explicit reconfigure. Explicit source-revision overrides and source archives retain their documented identity boundaries.
- Correct relocatable pkg-config metadata for supported multi-level installation directories, and validate real pkg-config consumer compilation/linking instead of relying only on a version query.
- Corrected product CPU-budget units to a versioned single-core process-CPU/wall-time contract. Active collectors, evidence, budgets and approval consumers reject the old online-capacity-normalized schemas; historical evidence is retained unchanged and cannot be relabelled as new qualification. Audio-time ratios require measured audio duration and are not inferred from wall time.
- The `process-cpu-one-core-v1` hard cut uses runtime-soak, target evidence, qualification manifest and policy schema v3; resource budgets and DUT/cohort summaries/receipts v2; qualification gate results v4. Multi-thread CPU use is not clipped to 100%, and actual sampled thread counts are retained.
- Removed deprecated token-boost arithmetic from decoder path scoring and made nonzero trailing-blank requirements effective for immediate keywords. Updated the diagnostic shadow scorer and added extreme-value, interrupted-blank and real keyword-example regressions. These corrections require fresh runner qualification; unchanged model/pack bytes do not transfer old acoustic results.
- Hardened SDK release fixture preparation and test inventory, and bound bootstrap cleanup to successful publication and the exact triggering commit. Active-run retention checks no longer depend on the newest 100 historical runs.
- Expanded build-configuration fingerprints to bind configuration-specific flags, including Release optimization options. Configuration fingerprints remain distinct from the separately verified runner binary SHA.
- Added offline durable-trace pointer identity and corruption checks, bounded installed-distribution metadata reads, and a single current D20 preparation checklist. D20 execution admission remains false; these changes perform no new numerical experiment or acoustic qualification.
- Fixed decoder normalization for large common finite logit offsets by retaining the maximum separately from the log-sum correction. This preserves the exponential approximation and does not change model weights, keyword thresholds, KWSP v2 or KWKP v3 layouts.
- Ordinary-range confidence values can still change by floating-point rounding, including decisions extremely close to a threshold. Unchanged weights and thresholds do **not** make historical acoustic results transferable to the new runner: qualify the new source/runner independently before any shipping claim. Historical FAIL results and frozen candidates remain bound to their original commit and runner; this fix does not retroactively approve or rewrite them.
- Corrected keyword-pack ownership documentation: decoded tokens live inside the opened pack, which must not be copied or moved without rebinding. Reopen the blob into a new destination for an independent pack. Engine keyword setters copy configuration, so successfully installed packs need not outlive the engine.

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
