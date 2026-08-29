# Changelog

## Unreleased

- Add versioned discontinuity/external-AFE metadata and v2 runtime telemetry.
- Add exact build identity and dirty-source marking.
- Require shipping-approved SKU policies and byte-bound collector/raw/attestation evidence.
- Add runtime source/config identity to target board benchmark output.
- Restore Python 3.8 compatibility for synthetic domain generation.

All notable source-level changes are recorded here. A source/software version does **not** imply that a particular Mandarin wake-word SKU has passed acoustic or target-board qualification.

## 0.2.0 software terminal baseline — 2026-08-29

### Added

- `KWKP` ABI v3 keyword records with per-keyword `min_trailing_blanks`, priority, `immediate` / `longest` / `grace` prefix policy and grace frames.
- Deterministic shared-prefix arbitration for overlapping phrases such as `小窝` / `小窝小窝`.
- Model-bound dual frontend support: `logmel` and `pcen-lite`, implemented across C runtime, dependency-free reference, training/export and provenance.
- Frontend-spec v2 release lineage and runtime↔model-lineage frontend identity gate, including tamper-negative qualification coverage.
- Acoustic scene renderer for near/mid/far distance, azimuth, RT60, SNR, white/fan/motor/media noise, playback/AEC residual and proxy/command AFE adapters.
- Measured dual-mic RIR manifest ingestion with deterministic convolution, RIR SHA256/onset/ITD evidence and direct integration into domain rendering.
- Command-AFE result sidecar with reported latency and stable provenance over command template, executable/config/input/output/result bytes.
- Domain metric scorer with distance/azimuth/RT60/noise/playback/composite buckets, negative-only FAR exposure and global monotonic one-to-one keyword confusion assignment.
- Multi-dimensional EMA curriculum over distance, azimuth, RT60, noise, playback and composite hard domains.
- Deterministic `far -> mid -> near` positive rotation for calibration/test/qualification synthetic renders so far-field evidence cannot disappear through random sampling.
- Raw streaming runtime tool plus frozen-model, multi-shard long-FAR nightly aggregation.
- One-sided statistical qualification bounds: Wilson upper bound for FRR and exact Poisson upper bound for FAR/hour.
- Identity-aware dataset audit with hard speaker/session/source isolation and optional room/device isolation policy.
- Training environment lock/Dockerfile and checkpoint/model provenance for Python, PyTorch, platform, training-code hash, lock/Dockerfile hash and optional container digest.
- Zero-I/O runtime telemetry through `kws_engine_get_stats()`.
- Cortex-A32 NEON int8-weight × float-activation GEMV while retaining the scalar portable reference path.
- Real-artifact `kws_board_bench` for target Linux boards, reporting mean/p50/p95/p99/max processing time, RTF, artifact/arena bytes and p99 scheduling headroom.
- Tag-driven release workflow with full hosted/sanitizer/fuzz/Cortex-A32 gates, deterministic SDK/source archives, SHA256SUMS, SPDX SBOM and GitHub artifact/SBOM attestations.
- Evaluation provenance sidecars and byte-complete release qualification binding actual model/checkpoint/training tokens/manifests/pack/eval/board artifacts.
- Clang libFuzzer + ASan/UBSan parser smoke for `.kwm` and `.kwk`.

### Changed

- SDK/package version is `0.2.0`; CMake package compatibility is `ExactVersion` during the hard-cut 0.x phase.
- Artifact versioning is explicit: model remains `KWSP` ABI v2; keyword pack is `KWKP` ABI v3.
- Decoder keeps independent nonblank/blank-separated Viterbi states for repeated-token CTC semantics and bounded pending state for shared-prefix policy.
- Training requires the exact token vocabulary instead of the removed size-only `--vocab-size` interface.
- Warm start/export preserve vocabulary, geometry, frontend and training-environment identity.
- PyTorch/reference/C frontends share the same fixed feature contract; CI compares both logmel and PCEN-lite paths.
- PCEN removes avoidable generic `powf` work by using `sqrtf` and a precomputed delta-root constant while retaining frontend parity gates.
- Hosted performance regression now covers logmel/1-keyword, logmel/4-keyword, PCEN/4-keyword and PCEN/16-keyword shared-prefix worst-case configurations.
- Domain qualification keeps far-field and negative-only hard-domain evidence strict instead of treating absent evidence as success.
- Nightly FAR exposure is aggregated only across shards with the exact same runner/model/keyword-pack identity; training-seed robustness is not counted as single-model exposure.
- Qualification policy schema v2 requires both point-estimate thresholds and statistical confidence upper bounds.
- Release gate requires model ABI v2, keyword-pack ABI v3, frontend-spec v2 and exact runtime/model-lineage frontend identity.
- Documentation distinguishes synthetic-domain/measured-RIR/software evidence from real 0.3–5 m human/device shipping qualification.

### Current evidence boundary

Hosted CI exercises GCC/Clang, ASan/UBSan, libFuzzer, Cortex-A32 ARMv7 hard-float cross-build, multi-domain/multi-frontend iteration, frozen-model long-FAR regression and byte-complete release qualification. These are software/synthetic/measured-RIR-capable regression mechanisms only. Real Mandarin held-out recordings, shipping microphone/enclosure/audio-pipeline qualification and physical target-board CPU/RSS/stack/thermal/power/soak evidence remain Issue #2 gates.

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
