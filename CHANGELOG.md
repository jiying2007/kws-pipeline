# Changelog

All notable source-level changes are recorded here. A source/software version does **not** imply that a particular Mandarin wake-word SKU has passed acoustic or target-board qualification.

## Unreleased

### Added

- `KWKP` ABI v3 keyword records with per-keyword `min_trailing_blanks`, priority, `immediate` / `longest` / `grace` prefix policy and grace frames.
- Deterministic shared-prefix arbitration for phrases such as a short wake phrase and its longer extension.
- Model-bound dual frontend support: `logmel` and `pcen-lite`, implemented across C runtime, dependency-free reference, training/export and provenance.
- Frontend-spec v2 release lineage and runtime↔model-lineage frontend identity gate, including a tamper-negative qualification test.
- Acoustic scene renderer for nominal near/mid/far distance, azimuth, RT60, SNR, white/fan/motor/media noise, local playback/AEC residual and proxy/command AFE adapters.
- Domain metric scorer with distance/angle/RT60/noise/playback/composite buckets, keyword confusion and deterministic worst-domain score.
- Adaptive distance-domain curriculum and `iterate_domain.py` multi-frontend candidate selection through the real C runtime.
- Deterministic `far -> mid -> near` positive rotation for calibration/test/qualification synthetic domain renders, preventing far-field gates from depending on random coverage; per-split distance histograms are retained.
- Raw streaming runtime tool, long continuous FAR regression runner and nightly hosted FAR workflow.
- Real-artifact `kws_board_bench` for target Linux boards, reporting mean/p50/p95/p99/max processing time, RTF, artifact/arena bytes and p99 scheduling headroom.
- Evaluation provenance sidecars and byte-complete release qualification binding actual model/checkpoint/training tokens/manifests/pack/eval/board artifacts.
- Dataset split auditor using decoded mono-16-kHz PCM16 SHA256.
- Clang libFuzzer + ASan/UBSan parser smoke for `.kwm` and `.kwk`.

### Changed

- Artifact versioning is explicit: model remains `KWSP` ABI v2; keyword pack is hard-cut to `KWKP` ABI v3.
- Decoder keeps independent nonblank/blank-separated Viterbi states for repeated-token CTC semantics and bounded pending state for shared-prefix policy.
- Training requires the exact token vocabulary instead of the removed size-only `--vocab-size` interface.
- Warm start/export now preserve vocabulary, geometry and frontend identity.
- PyTorch/reference/C frontends share the same fixed feature contract; CI compares both logmel and PCEN-lite paths.
- Domain qualification keeps the far-field gate strict and fixes data generation instead of treating a missing far domain as success.
- Release gate requires model ABI v2, keyword-pack ABI v3, frontend-spec v2 and exact runtime/model-lineage frontend identity.
- Documentation now distinguishes synthetic-domain/hosted evidence from real 3–5 m human/device qualification.

### Current evidence boundary

Hosted CI exercises GCC/Clang, ASan/UBSan, libFuzzer, Cortex-A32 ARMv7 cross-build, domain-aware multi-frontend iteration, long-FAR smoke and byte-complete release qualification. These are software/synthetic regression signals only. Real Mandarin held-out recordings and physical target-board evidence remain issue #2 gates.

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

The source baseline is not a claim that a real Mandarin base model or target-board acoustic qualification has been completed. That evidence is tracked separately in issue #2.
