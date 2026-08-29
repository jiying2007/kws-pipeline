# Changelog

All notable source-level changes are recorded here. A source/software version does **not** imply that a particular Mandarin wake-word SKU has passed acoustic or target-board qualification.

## Unreleased

### Added

- Real-artifact `kws_board_bench` for target Linux boards, reporting mean/p50/p95/p99/max processing time, RTF, artifact/arena bytes and p99 scheduling headroom.
- Dependency-free C SHA256 support so board results self-identify the exact benchmark runner, model, keyword pack and board audio.
- Evaluation provenance sidecars binding the exact evaluation runner/model/keyword-pack/references/detections by SHA256.
- Deterministic `qualification_manifest.py` that independently re-hashes all selected evidence files, revalidates canonical ABI-v2 model/pack/config/vocabulary identity, recomputes corpus and board-WAV statistics, and verifies acoustic/board formulas and cross-artifact hashes.
- `qualification_gate.py` with an explicit external SKU policy, manifest/policy hash binding, strict type checking, and pass/fail/invalid exit semantics.
- Policy gates for FAR/FRR/latency/runtime headroom plus soak, CPU, RSS, stack high-water mark, maximum temperature and average power.
- Dependency-free frontend feature specification plus `kws_feature_dump`; GCC/Clang CI compares multiple real C frontend fixtures against the reference with a tight numerical tolerance.
- Dataset split auditor that detects leakage by decoded mono-16-kHz PCM16 SHA256 while retaining WAV-container hashes for provenance, including rewrapped/renamed copies.
- Direct decoder contracts for adjacent repeated CTC tokens.
- Clang libFuzzer + ASan/UBSan parser smoke targets for `.kwm` and `.kwk` using canonical corpus seeds.
- Target-board evidence and qualification-policy templates.
- Release-qualification, contribution and security documentation.

### Changed

- The keyword Trie now keeps independent nonblank and blank-separated Viterbi scores. Adjacent identical acoustic tokens can advance only from a blank-separated state, matching the structural CTC repeated-label rule without introducing a general prefix beam.
- Training now requires the exact token vocabulary instead of a size-only `--vocab-size`. Checkpoints retain vocabulary fingerprint/token SHA256, training-manifest hashes, frontend-spec version, seed and optimizer settings; warm starts verify the same identity.
- Model export refuses to bind checkpoint weights to a same-sized but differently mapped token vocabulary.
- PyTorch frontend geometry imports the same FFT/mel/scale constants as the dependency-free frontend spec.
- Continuous-audio metric summaries include SHA256 for the exact reference and detection inputs.
- Hosted tool WAV/file helpers are shared by `kws_wav` and board qualification tooling.
- Performance documentation distinguishes hosted regression signals from artifact-bound shipping evidence.
- The earlier weaker release-manifest path was removed; `qualification_manifest.py` is the single supported qualification-manifest generator.

### Current hosted regression signal

On the GitHub GCC runner used by CI #208, the default 32x48x420 benchmark reported `model_bytes=25944`, `engine_bytes=17520` and RTF `0.001447`. The larger engine arena reflects the independent blank/nonblank Trie states. These hosted numbers are regression signals only, not Cortex-A32 measurements.

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

The 0.1.0 source baseline is not a claim that a real Mandarin base model or target-board acoustic qualification has been completed. That evidence is tracked separately in issue #2.
