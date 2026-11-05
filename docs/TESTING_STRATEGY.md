# Research testing strategy

This repository studies Mandarin keyword spotting and an embeddable low-resource C runtime; it is **not** a product release or certification program. Keep distinct evidence classes:

- **Source correctness:** strict GCC/Clang, CTest, frontend/model/decoder parity, deterministic tests and API contracts.
- **Input safety:** ASan/UBSan, parser fuzzing, invalid-audio and malformed-model rejection.
- **ISA feasibility:** Cortex-A32 ARMv7 hard-float cross-build and QEMU execution; neither is a measured SSC305 performance result.
- **Training reproducibility:** genuine PyTorch CTC training, frozen split/source manifests, export/quantize/runtime readback and repeatable optimizer-budget comparison.
- **Development stress testing:** synthetic/TTS/cloned-voice and longstream negatives, with actual audio exposure, provenance and measured FAR/FRR; never call these independent human truth.
- **Independent acoustic evidence:** reserved natural speech, accents, near-homophones, room noise and playback; report FAR/hour, FRR, latency and confidence bounds with held-out isolation.
- **Optional physical-target measurements:** source-bound actual board timings, CPU/RSS/I/O, power/thermal and capture continuity; absent evidence is `NOT_RUN`.

A successful hosted/CI experiment proves only its stated evidence class. No GitHub Release tag, SKU/shipping approval, arbitrary 72-hour soak or commercial qualification is a scientific prerequisite. Applicable audio/dataset licenses, speaker consent and separation of calibration from independent tests remain required.

New `tests/test_*.py` files must be discoverable by the CI inventory, including tests run through a dedicated CI step. Static analysis and coverage remain source regressions, not substitutes for independent acoustic or hardware measurements.
