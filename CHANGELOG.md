# Changelog

All notable source-level changes are recorded here. A source/software version does **not** imply that a particular Mandarin wake-word SKU has passed acoustic or target-board qualification.

## Unreleased

### Added

- Real-artifact `kws_board_bench` for target Linux boards, reporting mean/p50/p95/p99/max processing time, RTF, artifact/arena bytes and p99 scheduling headroom.
- Evaluation provenance sidecars binding runner/model/keyword-pack/references/detections by SHA256.
- Deterministic `release_manifest.py` qualification manifest covering artifact identity, acoustic results, board performance and target evidence.
- `qualification_gate.py` with an explicit external SKU policy and pass/fail/invalid exit semantics.
- Target-board evidence and qualification-policy templates.
- Release-qualification, contribution and security documentation.
- CI coverage for corpus provenance, real-artifact board-benchmark contract, qualification manifest/policy gate, and Cortex-A32 cross-build of target qualification tools.

### Changed

- Continuous-audio metric summaries now include SHA256 for the exact reference and detection inputs.
- Hosted tool WAV/file/JSON helpers are shared by `kws_wav` and board qualification tooling.
- Performance documentation now distinguishes hosted regression signals from artifact-bound shipping evidence.

## 0.1.0 source baseline — 2026-08-29

First complete merged software baseline:

- fixed-memory C11 always-on KWS runtime;
- 16-kHz / 25-ms / 20-ms-hop frontend and tiny int8-weight streaming recurrent acoustic model;
- ABI-v2 `.kwm` and field-updatable `.kwk` formats bound to a 64-bit vocabulary fingerprint;
- shared-prefix keyword trie and configurable Mandarin/pinyin wake phrases such as `你好小窝` / `小窝小窝`;
- L0 keyword update, L1 calibration/hard-negative workflow and L2 `--head-only` shallow customization;
- CTC training/export reference toolchain;
- real C WAV corpus runner, continuous-audio FAR/hour + FRR + latency scoring and hard-negative mining;
- strict GCC/Clang, CTest, ASan/UBSan and Cortex-A32 ARMv7 hard-float CI;
- CMake/pkg-config SDK packaging and clean consumer test;
- English and Simplified Chinese documentation.

The 0.1.0 source baseline is not a claim that a real Mandarin base model or target-board acoustic qualification has been completed. That evidence is tracked separately.
