# Isolated full-donor FSMN research components

Research-only C11 components. The fixed original2599-output implementation **failed strict intermediate/final-logit numerical gates**. Subsequent fixed PCM diagnostics preserved probability tolerances and observed non-score event fields, with small score differences. This does not establish product qualification, FAR or SSC305 performance.

## Components

- `pcm.c/h`: canonical4800-sample grouping, one actual short tail, original800-sample gate, retained waveform, no EOF padding/flush; composes the separately maintained donor_fbank frontend
- `splice.c/h`: exact feature-level ±2context/skip3 phase, short-tail/reset/overflow contracts
- `fsmn.c/h`: fixed four-way FP32 full2599 kernel, original finite-memory indexing, caller-owned weights/state, no allocation or file I/O in model step, finite faults requiring reset
- `manifest.py`: local Python research loader with fixed donor payload identity, exact30-tensor schema/hash/layout/finite checks. This is not a standalone C serialized-file/hash loader
- `tests/`: portable synthetic tests plus historical/reproducibility drivers. Donor-dependent drivers require the explicitly pinned external local research inputs and are not silently downloaded or run by CI

Parameter origin is the separately obtained official `iic/speech_charctc_kws_phone-xiaoyun` checkpoint. No checkpoint, exported payload, large donor-derived NPZ, corpus audio or runtime dependency is included. See NOTICE.md and LICENSE.wekws. The existing product frontend/model remain untouched.

## Tests

Portable CI requires only Python standard library, a C11 compiler and libm:

`python research/donor_fsmn/tests/run_portable.py`

It uses self-created exact-power-of-two synthetic coefficients, not pretrained tensors. It verifies delayed cache/tap semantics, no unintended ReLU between input/output affine pairs, reset/invalid model handling and canonical PCM grouping. `tests/sanitize_smoke.c` adds finite-overflow/fault checks. These passing tests are not a claim that the failed donor numerical gates pass.

The executed source-bound reference/comparison/profiling drivers retain their historical `/workspace/shared` input defaults to preserve their recorded code hashes. They do not constitute a portable dependency installer. Explicit separate setup/review is required before rerunning any experiment, particularly scripts that refuse to overwrite their original output paths.

## Authoritative evidence destination

Scalar/per-record experiment evidence and failure chronology are retained in the [immutable kws-data evidence snapshot](https://github.com/jiying2007/kws-data/tree/d9ed60533332246292ff0f71fc4669f83656ccc1/research/2026-09-30-full-fsmn-numeric-alignment). This is the reviewed research commit from draft PR8, not a claimed merge commit. Code remains here; large local arrays and provider tensors are excluded from both archives.

Evidence includes the initial/reference-generator failures, old B failures, the analytically consistent but uninformatively huge propagation envelope, real-PCM fbank/logit failures, retained observed behavior and the fixed x86 Python/Torch/ctypes profile. No result silently replaces an earlier failure. The measured process is not all-C end-to-end, and host cost cannot be extrapolated to SSC305.
