# Changelog

## Unreleased

- Add versioned discontinuity/external-AFE metadata and v2 runtime telemetry.
- Add exact build identity and dirty-source marking.
- Require shipping-approved SKU policies and byte-bound collector/raw/attestation evidence.
- Add runtime source/config identity to target board benchmark output.
- Restore Python 3.8 compatibility for synthetic domain generation.

All notable source-level changes are recorded here. A source/software version does **not** imply that a particular Mandarin wake-word SKU has passed acoustic or target-board qualification.

## 0.3.0 software evidence hardening — 2026-08-29

### Added

- Canonical training/evaluation corpus identity with per-WAV file SHA256, decoded mono-16-kHz PCM16 SHA256, frame count/duration and whole-corpus SHA256.
- TSV + schema-rich JSONL training support, including speaker/session/source/room/device metadata for human qualification datasets.
- Model provenance **schema v3** binding the real training corpus rather than only training-manifest bytes.
- Evaluation provenance **schema v2** binding every real held-out WAV and rejecting `references.duration_s` that disagrees with actual WAV duration.
- Qualification manifest **schema v2** that reopens/re-hashes original training/evaluation WAVs and validates corpus identity instead of trusting provenance declarations.
- Qualification gate result **schema v3** for the new evidence contract.
- Mandatory clean dataset-audit artifact in qualification, including required speaker/session/source metadata coverage for final human data.
- Machine target-evidence collector **schema v2** with collector identity, machine/kernel/governor/resource fields and retained raw evidence SHA256.
- Raw power/evidence file binding plus instrument/calibration identity for external measurements.
- `kws_engine_notify_discontinuity()` and explicit XRUN/route/clock/suspend-resume reasons; discontinuity telemetry is retained without joining acoustic state across missing samples.
- Immutable-OCI training wrapper: `training/Dockerfile` no longer resolves/upgrades dependencies from the network and requires a digest-pinned prebuilt training base.
- `training/build_container.py` for controlled wrapper-image construction and validation.
- Weekly/manual production `torch_ctc` integration workflow driven by a digest-pinned repository training-image variable.
- C line-coverage gate and Clang static analyzer in CI/release matrices.
- Python test-inventory gate so new `tests/test_*.py` cannot silently remain outside official workflows.
- Independent two-build SDK byte-for-byte reproducibility gate.
- New terminal-hardening documentation for corpus identity, audio discontinuity, target evidence, reproducibility, testing strategy and governance target.

### Changed

- SDK/package version is **0.3.0** while deployable model/keyword artifacts remain **KWSP ABI v2 + KWKP ABI v3**.
- Release workflow now requires hosted GCC/Clang, coverage, Clang static analysis, ASan/UBSan, parser fuzzing, Cortex-A32 cross-build, new evidence-contract tests and reproducible SDK output before publishing.
- Shipping training checkpoints are expected to record the final immutable training image digest.
- `training/requirements.lock` defines the expected prebuilt environment rather than acting as a network-resolved pip installation lock.
- Release qualification now requires `--dataset-audit`, `--eval-audio-root`, `--evidence-collector` and one or more `--raw-evidence` inputs.
- Final FAR exposure is derived from the real evaluation WAV frame count and cannot be inflated by a reference-only duration declaration.
- Target CPU/RSS/stack/thermal/power/soak evidence is treated as physical evidence only when connected to a retained collector/raw measurement identity.
- Repository governance documentation distinguishes current single-maintainer incubation policy from the stricter terminal branch/ruleset target.
- English and Simplified Chinese README plus evaluation/integration/performance/security/contributing documentation now describe the v0.3 evidence contracts.

### Evidence boundary

v0.3 closes the previously identified **software evidence integrity** gaps: unchanged manifests can no longer hide replaced WAV bytes, held-out duration must match real audio, target evidence has a machine/raw provenance path, discontinuities have explicit runtime semantics, and release output is coverage/static-analysis/reproducibility gated.

It still does **not** manufacture final product evidence. Real Mandarin held-out speakers through the shipping microphone/enclosure/audio-pipeline, genuine 0.3–5 m acoustic coverage and physical Cortex-A32 CPU/RSS/stack/thermal/power/soak measurements remain Issue #2 gates.

## 0.2.0 software baseline — 2026-08-29

### Added

- `KWKP` ABI v3 keyword records with per-keyword `min_trailing_blanks`, priority, `immediate` / `longest` / `grace` prefix policy and grace frames.
- Deterministic shared-prefix arbitration for overlapping phrases such as `小窝` / `小窝小窝`.
- Model-bound dual frontend support: `logmel` and `pcen-lite`, implemented across C runtime, dependency-free reference, training/export and provenance.
- Frontend-spec v2 release lineage and runtime↔model-lineage frontend identity gate.
- Acoustic scene renderer for near/mid/far distance, azimuth, RT60, SNR, white/fan/motor/media noise, playback/AEC residual and proxy/command AFE adapters.
- Measured dual-mic RIR manifest ingestion with deterministic convolution and RIR identity evidence.
- Domain metric scorer and multi-dimensional adaptive curriculum.
- Deterministic `far -> mid -> near` positive rotation for synthetic calibration/test/qualification.
- Raw streaming runtime tool plus frozen-model, multi-shard long-FAR nightly aggregation.
- One-sided statistical qualification bounds: Wilson upper bound for FRR and exact Poisson upper bound for FAR/hour.
- Identity-aware dataset audit with hard speaker/session/source isolation and optional room/device isolation policy.
- Training environment/checkpoint/model provenance for Python, PyTorch, platform, training-code/lock/Dockerfile hashes and optional container digest.
- Zero-I/O runtime telemetry through `kws_engine_get_stats()`.
- Cortex-A32 NEON int8-weight × float-activation GEMV while retaining the scalar portable path.
- Real-artifact `kws_board_bench` for target Linux boards.
- Tag-driven release workflow with deterministic SDK/source archives, SHA256SUMS, SPDX SBOM and GitHub attestations.
- Evaluation provenance and release-qualification cross-linking for model/checkpoint/tokens/manifests/pack/eval/board artifacts.
- Clang libFuzzer + ASan/UBSan parser smoke for `.kwm` and `.kwk`.

### Changed

- SDK/package version is `0.2.0`; CMake package compatibility is `ExactVersion` during the hard-cut 0.x phase.
- Artifact versioning is explicit: model remains `KWSP` ABI v2; keyword pack is `KWKP` ABI v3.
- Decoder keeps independent nonblank/blank-separated Viterbi states for repeated-token CTC semantics and bounded pending state for shared-prefix policy.
- Training requires the exact token vocabulary instead of the removed size-only interface.
- Warm start/export preserve vocabulary, geometry, frontend and training-environment identity.
- PyTorch/reference/C frontends share the same fixed feature contract.
- Hosted performance regression covers normal and maximum keyword/frontend envelopes.
- Domain qualification keeps far-field and negative-only hard-domain evidence strict.
- Nightly FAR exposure aggregates only identical runner/model/pack tuples.
- Qualification policy schema v2 requires both point-estimate and statistical confidence upper-bound thresholds.
- Documentation distinguishes synthetic/measured-RIR/software evidence from real 0.3–5 m product qualification.

### Evidence boundary

The v0.2 repository proved software/runtime/synthetic/release-integrity mechanisms but did not fully bind original training/evaluation audio bytes or machine/raw target evidence. Those gaps are closed by v0.3; physical/human product qualification remains external.

## 0.1.0 source baseline — 2026-08-29

First complete merged software baseline:

- fixed-memory C11 always-on KWS runtime;
- 16-kHz / 25-ms / 20-ms-hop frontend and tiny int8-weight streaming recurrent acoustic model;
- `KWSP` ABI v2 model format and vocabulary-bound field-updatable keyword packs;
- shared-prefix keyword trie and configurable Mandarin/pinyin wake phrases;
- L0 keyword update, L1 calibration/hard-negative workflow and L2 `--head-only` shallow customization;
- CTC training/export reference toolchain;
- real C WAV corpus runner, continuous-audio FAR/hour + FRR + latency scoring;
- strict GCC/Clang, CTest, ASan/UBSan and Cortex-A32 ARMv7 hard-float CI;
- CMake/pkg-config SDK packaging and clean consumer test;
- English and Simplified Chinese documentation.

The source baseline is not a claim that a real Mandarin base model or target-board acoustic qualification has been completed. That evidence is tracked separately in Issue #2.
