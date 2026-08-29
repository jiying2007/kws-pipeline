# Testing strategy

The project keeps separate evidence classes:

- per-change hosted correctness: strict GCC/Clang, CTest, frontend parity and deterministic offline tests;
- memory/input hardening: ASan/UBSan and parser fuzzing;
- ISA compatibility: Cortex-A32 ARMv7 hard-float cross-build;
- lower-frequency production-path integration: real PyTorch CTC train/export/quantize/runtime loop;
- recurring state-regression evidence: frozen-model long-FAR streaming;
- release qualification: artifact/corpus/target evidence gate;
- external shipping evidence: real human/device acoustic and physical-board qualification.

New `tests/test_*.py` files must be discoverable by CI inventory even when a test needs a dedicated runner invocation. Coverage/static-analysis signals should be regression gates, not substitutes for product evidence.
